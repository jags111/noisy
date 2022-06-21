#!/usr/bin/env python3
'''Command Line Interface (CLI) for Noisy. To instantiate a new model, train
and finally sample from it, run the following commands:

    ./cli.py init  # Creates a new model
    ./cli.py train  # Logs to wandb.ai, interrupt with Ctrl+c
    ./cli.py syn  # Samples from the model

Each command will print relevant information to stdout. For more options, add
the --help flag to any command.
'''
from typing import Optional
import logging
from pathlib import Path
import click
import torch
import math

import noisy


logger = logging.getLogger('noisy.cli')
DEFAULT_ZOO = Path('./zoo/')
DEFAULT_CONFIG = Path('./configs/default.yaml')


def _float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    assert isinstance(x, str)
    ctx = math.__dict__
    return float(eval(x, ctx))


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
    fmt = '[%(asctime)s|%(name)s|%(levelname)s] %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)


@cli.command('init')
@click.option('--workdir', '-w', type=Path, default=None)
@click.option('--config', '-c', type=Path, default=None)
@click.option('--force', '-f', type=bool, default=False)
@click.option('--wandb-name', '-n', type=str, default=None)
def init(workdir: Optional[Path], config: Optional[Path], force: bool,
         wandb_name: Optional[str]) -> None:
    '''Instantiate a new, untrained model.'''
    if workdir is None:
        workdir = DEFAULT_ZOO / noisy.utils.random_name()
    if config is None:
        config = DEFAULT_CONFIG
    cfg = noisy.utils.AttrDict.from_yaml(config)
    if wandb_name is not None:
        cfg.wandb.name = wandb_name
    logger.info(f'Initializing new run in: {workdir}, using the configuration '
                f'in: {config}')
    cp = noisy.workdir.init(workdir, cfg, force=force)
    noisy.utils.make_symlink(noisy.workdir.LATEST_PROJ_SL, workdir)
    noisy.utils.make_symlink(workdir / noisy.workdir.LATEST_CP_SL, cp)


@cli.command('train')
@click.option('--workdir', '-w', type=Path, default=None)
@click.option('--checkpoint', '-c', type=Path, default=None)
@click.option('--device', '-d', type=str, default=None)
@click.option('--no-wandb', '-s', type=bool, default=False)
def train(workdir: Optional[Path], checkpoint: Optional[Path],
          device: Optional[str], no_wandb: bool) -> None:
    '''Train the model in the specified checkpoint. If no checkpoint is
    specified, default to the most recent checkpoint.'''
    if checkpoint is None:
        checkpoint = noisy.workdir.LATEST_PROJ_SL / noisy.workdir.LATEST_CP_SL
    if workdir is None:
        workdir = checkpoint.parent
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_ = torch.device(device)
    cfg, model, optim, ctx = noisy.workdir.load_checkpoint(checkpoint, device_)
    if no_wandb:
        cfg.wandb.enable = False
    relpath = noisy.utils.rel_path(checkpoint.resolve())
    logger.info(f'Loaded run from: {relpath}')
    ds = noisy.dataset.ImgDataset(cfg)
    noisy.training.train(cfg, model, optim, ds, ctx, workdir, device_)


@cli.command('syn')
@click.option('--number', '-n', type=_int, default=1,
              help='Number of images to sample')
@click.option('--iters', '-i', type=_int, default=None)
@click.option('--show-intermediate', type=_int, default=None,
              help='Show the intermediate result every n steps')
@click.option('--std', type=_float, default=None,
              help='Scaling factor of each model invocation')
@click.option('--checkpoint', '-c', type=Path, default=None)
@click.option('--device', '-d', type=str, default=None)
def syn(number: int, iters: Optional[int], show_intermediate: Optional[int],
        std: Optional[float], checkpoint: Optional[Path], device: Optional[str]
        ) -> None:
    '''Sample from the model in the specified checkpoint. If no checkpoint is
    specified, default to the most recent checkpoint.'''
    if checkpoint is None:
        checkpoint = noisy.workdir.LATEST_PROJ_SL / noisy.workdir.LATEST_CP_SL
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Loading model from {checkpoint}...')
    device_ = torch.device(device)
    cfg = noisy.workdir.load_cfg(checkpoint)
    model = noisy.workdir.load_model(checkpoint, cfg)
    if show_intermediate is None:
        img = noisy.sampling.sample(model, cfg, number, iterations=iters,
                                    std=std, show_bar=True,
                                    device=device_)
    else:
        imgs = noisy.sampling.sample_iter(model, cfg, number,
                                          show_intermediate,
                                          iterations=iters, std=std,
                                          show_bar=True, device=device_)
        img = torch.cat([*imgs])
    img_norm = img * 0.5 + 0.5
    noisy.utils.show(img_norm, clip=True)


@cli.command('dev')
@click.option('--task', '-t', type=str, default='te')
def dev(task: str) -> None:
    '''For debugging and development, please ignore :)'''
    if task == 'diff':
        # Display diffusion
        x = torch.ones(10, 1, 32, 32)
        beta = torch.rand((10,))
        noise = torch.randn((10, 1, 32, 32))
        diff = noisy.training.diffuse(x, beta, noise, 'linear')
        print(beta)
        noisy.utils.show(diff)


if __name__ == '__main__':
    cli()
