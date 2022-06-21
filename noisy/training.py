from typing import Optional, Any, Union, Dict, Callable
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
from .sampling import sample
from . import workdir
from .perf import PerfMeasurer, get_totals as get_perf_totals


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


def diffuse(x: Tensor, beta: Union[float, Tensor], noise: Tensor,
            method_or_cfg: Union[str, AttrDict]) -> Tensor:
    '''Perform diffusion on the image. Intuitively: The larger the beta, the
    more noise.'''
    # Check beta
    if isinstance(beta, Tensor):
        assert 0. <= beta.min() and 1 >= beta.max(), beta
        b, = beta.shape
        assert x.size(0) == b
        beta = beta.reshape(-1, 1, 1, 1)
    else:
        assert 0 <= beta <= 1., f'beta should be in [0, 1], but is {beta}'
    # Check method
    if isinstance(method_or_cfg, str):
        method = method_or_cfg
    else:
        method = method_or_cfg.arch.diffusion_method
    methods: Dict[str, Callable[[], Tensor]] = {
        'sqrt': lambda: ((1 - beta) ** 0.5 * x  # type: ignore
                         + beta ** 0.5 * noise),
        'linear': lambda: (1 - beta) * x + beta * noise,
    }
    if method not in methods:
        raise ValueError(f'Unknown diffusion method: {method}, expected one of'
                         f' {list(methods.keys())}')
    # Execute
    x = methods[method]()
    assert isinstance(x, Tensor)
    return x


def train(cfg: AttrDict, model: Model, optim: AdamW, ds: Dataset[Tensor],
          ctx: TrainingContext, wd: Path, device: torch.device) -> None:
    dl = DataLoader(ds, cfg.training.batch_size, shuffle=True, pin_memory=True)
    batches = (batch for _ in count() for batch in dl)
    if cfg.wandb.enable:
        wandb_run = load_wandb_run(ctx, cfg, model)
    else:
        wandb_run = None
    for batch in batches:
        model.train().to(device)
        with PerfMeasurer('train.prep'):
            # (1) Prepare the data.
            batch = batch.to(device)
            batch_size = batch.size(0)
            noise = torch.randn(batch.shape, device=device)
            # Pick a random t for each sample in the batch
            t = torch.randint(cfg.T - cfg.training.steps - 1,
                              size=(batch_size,), device=device)
            # Add noise to the target..
            targets_beta = t / cfg.T
            target = diffuse(batch, targets_beta, noise, cfg)
            # ...and slightly stronger noise to the inputs.
            images_beta = (t + cfg.training.steps) / cfg.T
            images = diffuse(batch, images_beta, noise, cfg)
        with PerfMeasurer('train.run'):
            # (2) Pass the inputs to the model and iterate for
            # `cfg.training.steps` steps.
            model.zero_grad()
            for dt in range(cfg.training.steps):
                t_ = t - dt + cfg.training.steps
                images = images + model(images, t_ / cfg.T, std=1 / cfg.T)
            loss = F.mse_loss(images, target)
            loss.backward()  # type: ignore
            optim.step()
        if ctx.periodically(cfg.training.metrics_freq):
            with PerfMeasurer('train.metrics'):
                with torch.no_grad():
                    # (3) Update the EMA of the loss and do other housekeeping.
                    abs_loss = float(loss.detach().mean().sqrt().item())
                ctx.loss_ema = ema(abs_loss, ctx.loss_ema)
        # Advance the iteration counter.
        ctx.iteration += 1
        # Logging to stdout.
        if ctx.periodically(cfg.training.log_freq):
            logger.info(f'{ctx.iteration} | {ctx.loss_ema:5.7}')
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
        # Saving checkpoints.
        if ctx.periodically(cfg.training.checkpoint_freq):
            with PerfMeasurer('train.checkpoint'):
                cp = wd / str(ctx.iteration).zfill(6)
                workdir.save_checkpoint(cp, model, optim, cfg, ctx)
                make_symlink(wd / workdir.LATEST_CP_SL, cp)
                logger.info(f'Saved checkpoint to: {rel_path(cp)}')


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
    imgs = sample(model, cfg, number=8, show_bar=False, device=device)
    imgs_np = imgs.cpu().detach().numpy()
    imgs_np = imgs_np.transpose((0, 2, 3, 1))
    log_dict = {
        'Imgs': [wandb.Image(gen_img) for gen_img in imgs_np],
    }
    run.log(log_dict, step=ctx.iteration)
