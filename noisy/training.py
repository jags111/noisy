from typing import Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from logging import getLogger
from itertools import count
from pprint import pformat
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from .models import Model
from .utils import AttrDict, rel_path, make_symlink, random_name, ema
from . import workdir
from .perf import PerfMeasurer, get_totals as get_perf_totals
from .diffusion import get_beta_sched, get_alpha_cumprod_sched, sample


logger = getLogger('noisy.training')


@dataclass
class TrainingContext:
    '''Values for training that need to be persistently stored in checkpoints,
    apart from model parameters and the optimiser state etc.'''
    iteration: int = 0
    loss_ema: Optional[float] = None
    wandb_run_id: Optional[Any] = None

    def periodically(self, freq: int) -> bool:
        return self.iteration % freq == 0


def forward_diffuse(img: Tensor, t: Union[int, Tensor],
                    alpha_cumprod_sched: Tensor, noise: Optional[Tensor] = None
                    ) -> Tensor:
    if isinstance(t, int):
        t = torch.tensor([t] * img.size(0), device=img.device)
    if noise is None:
        noise = torch.randn_like(img)
    alpha_cumprods = alpha_cumprod_sched[t].reshape(-1, 1, 1, 1)
    img = (alpha_cumprods.sqrt() * img
           + (1 - alpha_cumprods).sqrt() * noise)
    return img


def train(cfg: AttrDict, model: Model, optim: AdamW, ds: Dataset[Tensor],
          ctx: TrainingContext, wd: Path, device: torch.device) -> None:
    dl = DataLoader(ds, cfg.training.batch_size, shuffle=True, pin_memory=True)
    batches = (batch for _ in count() for batch in dl)
    if cfg.wandb.enable:
        wandb_run = load_wandb_run(ctx, cfg, model)
    else:
        wandb_run = None
    beta_sched = get_beta_sched(cfg).to(device)
    alpha_cumprod_sched = get_alpha_cumprod_sched(beta_sched).to(device)

    def _save(cp: Optional[Path] = None) -> None:
        '''Saves the model etc. to a checkpoint. Extracted into a function to
        avoid code duplication.'''
        cp = cp or wd / str(ctx.iteration).zfill(6)
        workdir.save_checkpoint(cp, model, optim, cfg, ctx)
        make_symlink(wd / workdir.LATEST_CP_SL, cp)
        logger.info(f'Saved run to: {rel_path(cp)}')

    try:
        for batch in batches:
            model.train().to(device)
            with PerfMeasurer('train.prep'):
                # (1) Prepare the data.
                batch = batch.to(device)
                batch_size = batch.size(0)
                # Pick a random t for each sample in the batch
                t = torch.randint(cfg.diffusion.T, size=(batch_size,),
                                  device=device)
                alpha_cumprods = alpha_cumprod_sched[t].reshape(-1, 1, 1, 1)
                # Add noise to the target..
                noise = torch.randn_like(batch)
                inputs = (alpha_cumprods.sqrt() * batch
                          + (1 - alpha_cumprods).sqrt() * noise)
            with PerfMeasurer('train.model'):
                # (2) Pass the inputs to the model and iterate for
                # `cfg.training.steps` steps.
                model.zero_grad()
                estimate = model(inputs, t)
            with PerfMeasurer('train.grad'):
                loss = F.mse_loss(estimate, noise)
                loss.backward()  # type: ignore
            with PerfMeasurer('train.optim'):
                optim.step()
            if ctx.periodically(cfg.training.metrics_freq):
                with PerfMeasurer('train.metrics'):
                    with torch.no_grad():
                        # (3) Update the EMA of the loss and do other
                        # housekeeping.
                        abs_loss = float(loss.detach().mean().sqrt().item())
                    ctx.loss_ema = ema(abs_loss, ctx.loss_ema)
            # Advance the iteration counter.
            ctx.iteration += 1
            # Logging to stdout.
            if ctx.periodically(cfg.training.log_freq):
                logger.info(f'{ctx.iteration} | {ctx.loss_ema:5.7}')
            # Saving checkpoints.
            if ctx.periodically(cfg.training.checkpoint_freq):
                # Create a new checkpoint
                with PerfMeasurer('train.checkpoint'):
                    _save()
            if ctx.periodically(cfg.training.persist_freq):
                # Overwrite the 'current' checkpoint
                with PerfMeasurer('train.persist'):
                    _save(wd / 'current')
            # Logging to WandB.
            if cfg.wandb.enable and ctx.periodically(cfg.wandb.log_freq):
                with PerfMeasurer('train.wandb.log'):
                    assert wandb_run is not None
                    log_to_wandb(wandb_run, cfg, ctx)
            # Sending synthesised images to WandB.
            if cfg.wandb.enable and ctx.periodically(cfg.wandb.img_freq):
                with PerfMeasurer('train.wandb.img'):
                    assert wandb_run is not None
                    img_to_wandb(wandb_run, model, cfg, ctx, device)
    except KeyboardInterrupt:
        logger.info(f'Training manually ended at iteration {ctx.iteration}')
    finally:
        _save()


def load_wandb_run(ctx: TrainingContext, cfg: AttrDict, model: Model
                   ) -> WandbRun:
    if ctx.wandb_run_id is None:
        run = wandb.init(project=cfg.wandb.project,
                         group=cfg.wandb.group,
                         name=cfg.wandb.name or random_name(prefix='wandb'),
                         tags=cfg.wandb.tags,
                         resume=False,
                         notes=(f'Model:\n{model}\nConfig:\n{pformat(cfg)}'))
    else:
        run = wandb.init(id=ctx.wandb_run_id, resume=True)
    wandb.watch(model, log_freq=cfg.wandb.gradient_freq, idx=0, log='all')
    assert isinstance(run, WandbRun)
    ctx.wandb_run_id = run.id
    logger.info(f'Loaded WandB run with id: {run.id}, name: {run.name} and '
                f'URL: {run.get_url()}')
    return run


def log_to_wandb(run: WandbRun,
                 cfg: AttrDict,
                 ctx: TrainingContext) -> None:
    log_dict = {
        'Loss': ctx.loss_ema,
        'LR': cfg.optim.lr,
        'Batch Size': cfg.training.batch_size,
        **get_perf_totals(prefix='perf.')
    }
    run.log(log_dict, step=ctx.iteration)


def img_to_wandb(run: WandbRun,
                 model: Model,
                 cfg: AttrDict,
                 ctx: TrainingContext,
                 device: torch.device) -> None:
    imgs = sample(model, cfg, cfg.wandb.img_n, device=device,
                  show_bar=cfg.wandb.img_show_bar)
    imgs_np = imgs.cpu().detach().numpy()
    imgs_np = imgs_np.transpose((0, 2, 3, 1))
    log_dict = {
        'Imgs': [wandb.Image(gen_img) for gen_img in imgs_np],
    }
    run.log(log_dict, step=ctx.iteration)
