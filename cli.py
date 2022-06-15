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
def init(workdir: Optional[Path], config: Optional[Path], force: bool) -> None:
    if workdir is None:
        workdir = DEFAULT_ZOO / random_name()
    if config is None:
        config = DEFAULT_CONFIG
    logger.info(f'Initializing new run in: {workdir}, using the configuration '
                f'in: {config}')
    cp = noisy.workdir.init(workdir, config, force=force)
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


def random_name(prefix: str = 'run') -> str:
    counter_file = Path('.name_counter')
    if counter_file.exists():
        with open(counter_file, 'r') as fh:
            number = int(fh.read())
    else:
        number = 0
    with open(counter_file, 'w') as fh:
        fh.write(str(number + 1))
    return f'{prefix}-{str(number).zfill(4)}'


if __name__ == '__main__':
    cli()
