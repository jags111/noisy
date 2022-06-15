from typing import Optional
import random
from dataclasses import dataclass
from pathlib import Path
from logging import getLogger
from itertools import count
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from .models import Model
from .utils import AttrDict, rel_path, make_symlink
from . import workdir


logger = getLogger('noisy.training')


@dataclass
class TrainingContext:
    iteration: int = 0
    loss_ema: Optional[float] = None

    def periodically(self, freq: int) -> bool:
        return self.iteration % freq == 0


def diffuse(x: Tensor, beta: float, noise: Tensor) -> Tensor:
    x = (1 - beta) ** 0.5 * x + beta ** 0.5 * noise
    assert isinstance(x, Tensor)
    return x


def ema(x: float, acc: Optional[float], alpha: float = 0.95) -> float:
    if acc is None:
        return x
    return alpha * acc + (1 - alpha) * x


def train(cfg: AttrDict, model: Model, optim: AdamW, ds: Dataset[Tensor],
          ctx: TrainingContext, wd: Path, device: torch.device) -> None:
    model.to(device)
    dl = DataLoader(ds, cfg.training.batch_size)
    pairs = (batch for _ in count() for batch in dl)
    for batch in pairs:
        # Prepare the data
        batch = batch.to(device)
        noise = torch.randn(batch.shape, device=device)
        t = random.randint(0, cfg.T)
        # For now we just set beta to a constant
        inputs_beta = t / cfg.T
        targets_beta = (t + 1) / cfg.T
        inputs = diffuse(batch, inputs_beta, noise)
        target = diffuse(inputs, targets_beta - inputs_beta, noise)
        # The actual training
        model.zero_grad()
        estimation = model(target, t / cfg.T)
        # TODO: Try using the distance between the estimation and the line
        # between the batch and the inputs. See:
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        loss = F.mse_loss(estimation, target)
        loss.backward()  # type: ignore
        optim.step()
        # Update the EMA of the loss
        abs_loss = float(loss.mean().sqrt().item())
        ctx.loss_ema = ema(abs_loss, ctx.loss_ema)
        # Logging to stdout
        if ctx.periodically(cfg.training.log_freq):
            logger.info(f'{ctx.iteration} | {ctx.loss_ema:5.7}')
        # Saving checkpoints
        if ctx.periodically(cfg.training.checkpoint_freq):
            cp = wd / str(ctx.iteration).zfill(6)
            workdir.save_checkpoint(cp, model, optim, cfg, ctx)
            make_symlink(wd / workdir.LATEST_CP_SL, cp)
            logger.info(f'Saved checkpoint to: {rel_path(cp)}')
        ctx.iteration += 1
