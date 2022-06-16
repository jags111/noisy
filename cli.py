#!/usr/bin/env python3
from typing import Optional
import logging
from pathlib import Path
import click
import torch

import noisy


logger = logging.getLogger('noisy.cli')
DEFAULT_ZOO = Path('./zoo/')
DEFAULT_CONFIG = Path('./configs/default.yaml')


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
def train(workdir: Optional[Path], checkpoint: Optional[Path],
          device: Optional[str]) -> None:
    if checkpoint is None:
        checkpoint = noisy.workdir.LATEST_PROJ_SL / noisy.workdir.LATEST_CP_SL
    if workdir is None:
        workdir = checkpoint.parent
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_ = torch.device(device)
    cfg, model, optim, ctx = noisy.workdir.load_checkpoint(checkpoint, device_)
    relpath = noisy.utils.rel_path(checkpoint.resolve())
    logger.info(f'Loaded run from: {relpath}')
    ds = noisy.dataset.ImgDataset(cfg)
    noisy.training.train(cfg, model, optim, ds, ctx, workdir, device_)


@cli.command('syn')
@click.option('--number', '-n', type=int, default=1)
@click.option('--iters', '-i', type=int, default=None)
@click.option('--checkpoint', '-c', type=Path, default=None)
@click.option('--device', '-d', type=str, default=None)
def syn(number: int, iters: Optional[int], checkpoint: Optional[Path],
        device: Optional[str]) -> None:
    if checkpoint is None:
        checkpoint = noisy.workdir.LATEST_PROJ_SL / noisy.workdir.LATEST_CP_SL
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Loading model from {checkpoint}...')
    device_ = torch.device(device)
    cfg = noisy.workdir.load_cfg(checkpoint)
    model = noisy.workdir.load_model(checkpoint, cfg)
    img = noisy.synthesize.synthesize(model, cfg, number, iterations=iters,
                                      show_bar=True, device=device_)
    img_norm = img * 0.5 + 0.5
    noisy.utils.show(img_norm, clip=True)


@cli.command('dev')
@click.option('--task', '-t', type=str, default='te')
def dev(task: str) -> None:
    if task == 'te':
        # Display timestep embeddings
        x = torch.ones(10, 1, 32, 32)
        t = torch.rand((10,))
        emb = noisy.models.timestep_embedding(x, t, 1)
        print(t)
        noisy.utils.show(emb[:, :-1])
    elif task == 'diff':
        # Display diffusion
        x = torch.ones(10, 1, 32, 32)
        beta = torch.rand((10,))
        noise = torch.randn((10, 1, 32, 32))
        diff = noisy.training.diffuse(x, beta, noise)
        print(beta)
        noisy.utils.show(diff)


if __name__ == '__main__':
    cli()
