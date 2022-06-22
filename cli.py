#!/usr/bin/env python3
'''Command Line Interface (CLI) for Noisy. To instantiate a new model, train
and finally sample from it, run the following commands:

    ./cli.py init  # Creates a new model
    ./cli.py train  # Logs to wandb.ai, interrupt with Ctrl+c
    ./cli.py syn  # Samples from the model

Each command will print relevant information to stdout. For more options, add
the --help flag to any command.
'''
from typing import Optional, Union
import logging
from pathlib import Path
import click
import torch
import math
import os

import noisy


logger = logging.getLogger('noisy.cli')
DEFAULT_ZOO = Path('./zoo/')
DEFAULT_CONFIG = Path('./configs/default.yaml')


def _int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return int(x)
    assert isinstance(x, str), x
    ctx = math.__dict__
    return int(eval(x, ctx))


@click.group('cli')
def cli() -> None:
    '''The CLI for Noisy, a denoising diffusion probabilistic model.'''
    fmt = '[%(asctime)s|%(name)s|%(levelname)s] %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)


@cli.command('init')
@click.option('--workdir', '-w', type=Path, default=None,
              help='Path to the working directory.')
@click.option('--config', '-c', type=Path, default=None,
              help='Path to the configuration file.')
@click.option('--force', '-f', is_flag=True, default=False,
              help='Delete existing working directory.')
@click.option('--tmp', '-f', is_flag=True, default=False,
              help='Only for temporary projects, e.g. during debugging. '
              'Disables WandB.')
@click.option('--wandb-name', '-n', type=str, default=None,
              help='Overwrites the WandB name.')
def init(workdir: Optional[Path], config: Optional[Path], force: bool,
         tmp: bool, wandb_name: Optional[str]) -> None:
    '''Instantiate a new, untrained model inside a working directory and create
    an initial checkpoint.'''
    if config is None:
        config = DEFAULT_CONFIG
    cfg = noisy.utils.AttrDict.from_yaml(config)
    if tmp:
        workdir = DEFAULT_ZOO / 'tmp'
        force = True
        cfg.wandb.enable = False
    if workdir is None:
        workdir = DEFAULT_ZOO / noisy.utils.random_name()
    if wandb_name is not None:
        cfg.wandb.name = wandb_name
    logger.info(f'Initializing new run in: {workdir}, using the configuration '
                f'in: {config}')
    cp = noisy.workdir.init(workdir, cfg, force=force)
    noisy.utils.make_symlink(noisy.workdir.LATEST_PROJ_SL, workdir)
    noisy.utils.make_symlink(workdir / noisy.workdir.LATEST_CP_SL, cp)


@cli.command('train')
@click.option('--workdir', '-w', type=Path, default=None,
              help='Path to the working directory.')
@click.option('--checkpoint', '-c', type=Path, default=None,
              help='Path to the checkpoint to be loaded.')
@click.option('--device', '-d', type=click.Choice(('cpu', 'cuda')),
              default=None)
@click.option('--no-wandb', '-s', is_flag=True, default=False,
              help='Overwrite the wandb.enable flag in the configuration '
              'file.')
def train(workdir: Optional[Path], checkpoint: Optional[Path],
          device: Optional[str], no_wandb: bool) -> None:
    '''Load the model from the specified checkpoint and train it. If no
    checkpoint is specified, default to the most recent checkpoint.'''
    checkpoint = _ensure_cp(checkpoint)
    if workdir is None:
        workdir = checkpoint.parent
    device_ = _ensure_device(device)
    cfg, model, optim, ctx = noisy.workdir.load_checkpoint(checkpoint, device_)
    if no_wandb:
        cfg.wandb.enable = False
    logger.info(f'Loaded run from: {checkpoint}')
    ds = noisy.dataset.ImgDataset(cfg)
    noisy.training.train(cfg, model, optim, ds, ctx, workdir, device_)


@cli.command('sample')
@click.option('--number', '-n', type=_int, default=1,
              help='Number of images to sample')
@click.option('--checkpoint', '-c', type=Path, default=None,
              help='Path to the checkpoint to be loaded.')
@click.option('--device', '-d', type=click.Choice(('cpu', 'cuda')),
              default=None)
@click.option('--out', '-o', type=Path, default=None,
              help='File to save the samples image to. If None, the image will'
              ' be show.')
def sample(number: int, checkpoint: Optional[Path], device: Optional[str],
           out: Optional[Path]) -> None:
    '''Load the model from the specified checkpoint and sample from it. If no
    checkpoint is specified, default to the most recent checkpoint.'''
    checkpoint = _ensure_cp(checkpoint)
    device_ = _ensure_device(device)
    logger.info(f'Loading model from {checkpoint}...')
    cfg = noisy.workdir.load_cfg(checkpoint)
    model = noisy.workdir.load_model(checkpoint, cfg)
    img = noisy.diffusion.sample(model, cfg, number, device=device_,
                                 show_bar=True)
    img_norm = img * 0.5 + 0.5
    noisy.utils.show(img_norm, clip=True, out=out)


@cli.command('info')
@click.option('--checkpoint', '-c', type=Path, default=None,
              help='Path to the checkpoint to be loaded.')
def info(checkpoint: Optional[Path]) -> None:
    '''Print information regarding the specified checkpoint. If no checkpoint
    is specified, default to the most recent checkpoint.'''
    checkpoint = _ensure_cp(checkpoint)
    cfg = noisy.workdir.load_cfg(checkpoint)
    model = noisy.workdir.load_model(checkpoint, cfg)
    total_params = sum(p.numel() for p in model.parameters())
    cp_contents = (checkpoint / f for f in os.listdir(checkpoint))
    cp_size = sum(os.path.getsize(f) for f in cp_contents if os.path.isfile(f))
    ctx = noisy.workdir.load_ctx(checkpoint)
    # We do not log the information, as this would print it to stderr instead
    # of stdout.
    print(f'Checkpoint: {noisy.utils.rel_path(checkpoint.resolve())}')
    print(f'Iteration: {ctx.iteration}')
    print(f'Wandb ID: {ctx.wandb_run_id}')
    print(f'Loss EMA: {ctx.loss_ema}')
    print(f'Parameters: {total_params}')
    print(f'Checkpoint size: {round(cp_size / 1e6, 2)} MB')


@cli.command('dev')
@click.option('--task', '-t', type=str)
def dev(task: str) -> None:
    '''For debugging and development, please ignore :)'''
    pass


def _ensure_cp(checkpoint: Optional[Path]) -> Path:
    if checkpoint is None:
        checkpoint = noisy.workdir.LATEST_PROJ_SL / noisy.workdir.LATEST_CP_SL
        checkpoint = noisy.utils.rel_path(checkpoint.resolve())
    return checkpoint


def _ensure_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(device, str):
        return torch.device(device)
    return device


if __name__ == '__main__':
    cli()
