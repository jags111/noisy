#!/usr/bin/env python3
'''Command Line Interface (CLI) for Noisy. To instantiate a new model, train
and finally sample from it, run the following commands:

    ./cli.py init  # Creates a new model
    ./cli.py train  # Logs to wandb.ai, interrupt with Ctrl+c
    ./cli.py sample  # Samples from the model

Each command will print relevant information to stdout. For more options, add
the --help flag to any command.
'''
from typing import Optional, Union
import logging
from pathlib import Path
import click
import math
import os
import jinja2
import torch

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
@click.option('--no-symlink', is_flag=True)
def init(workdir: Optional[Path], config: Optional[Path], force: bool,
         tmp: bool, wandb_name: Optional[str], no_symlink: bool) -> None:
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
                f'in: {config} and model architecture "{cfg.arch.model}"')
    cp = noisy.workdir.init(workdir, cfg, force=force)
    if not no_symlink:
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
    cfg, model, optim, ctx, ema = noisy.workdir.load_checkpoint(checkpoint,
                                                                device_)
    if no_wandb:
        cfg.wandb.enable = False
    logger.info(f'Loaded run from: {checkpoint}')
    ds = noisy.dataset.ImgDataset(cfg)
    noisy.training.train(cfg, model, optim, ema, ds, ctx, workdir, device_)


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
@click.option('--no-ema', is_flag=True, default=False,
              help='Sample from the original model, not the EMA.')
@click.option('--no-bar', is_flag=True, default=False)
def sample(number: int, checkpoint: Optional[Path], device: Optional[str],
           out: Optional[Path], no_ema: bool, no_bar: bool) -> None:
    '''Load the model from the specified checkpoint and sample from it. If no
    checkpoint is specified, default to the most recent checkpoint.'''
    checkpoint = _ensure_cp(checkpoint)
    device_ = _ensure_device(device)
    logger.info(f'Loading model from {checkpoint}...')
    cfg = noisy.workdir.load_cfg(checkpoint)
    if no_ema:
        model = noisy.workdir.load_model(checkpoint, cfg)
    else:
        model = noisy.workdir.load_ema(checkpoint, cfg)
    img = noisy.diffusion.sample(model, cfg, number, device=device_,
                                 show_bar=not no_bar)
    img_norm = img * 0.5 + 0.5
    noisy.utils.show(img_norm, clip=True, out=out)


@cli.command('sample-blend')
@click.option('--number', '-n', type=_int, default=1,
              help='Number of images to sample')
@click.option('--blends', '-b', type=_int, default=32,
              help='Number of intermediate images')
@click.option('--checkpoint', '-c', type=Path, default=None,
              help='Path to the checkpoint to be loaded.')
@click.option('--device', '-d', type=click.Choice(('cpu', 'cuda')),
              default=None)
@click.option('--out', '-o', type=Path, default=None,
              help='File to save the samples image to. If None, the image will'
              ' be show.')
@click.option('--no-ema', is_flag=True, default=False,
              help='Sample from the original model, not the EMA.')
@click.option('--no-bar', is_flag=True, default=False)
def sample_blend(number: int, blends: int, checkpoint: Optional[Path],
                 device: Optional[str], out: Optional[Path], no_ema: bool,
                 no_bar: bool) -> None:
    '''Load the model from the specified checkpoint and sample from it, keeping
    track of intermediate results. If no checkpoint is specified, default to
    the most recent checkpoint.'''
    checkpoint = _ensure_cp(checkpoint)
    device_ = _ensure_device(device)
    logger.info(f'Loading model from {checkpoint}...')
    cfg = noisy.workdir.load_cfg(checkpoint)
    if no_ema:
        model = noisy.workdir.load_model(checkpoint, cfg)
    else:
        model = noisy.workdir.load_ema(checkpoint, cfg)
    img_it = noisy.diffusion.sample_iter(model, cfg, blends, number,
                                         device=device_, show_bar=not no_bar)
    img = torch.cat(list(img_it), dim=0)
    img_norm = img * 0.5 + 0.5
    n_rows = blends if number > 1 else 8
    noisy.utils.show_grid(img_norm, clip=True, out=out, n_rows=n_rows)


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
    ds = noisy.dataset.ImgDataset(cfg, lazy=True)
    ds_cache = ds.get_cache_path()
    ds_files = ds.get_files()
    # We do not log the information, as this would print it to stderr instead
    # of stdout.
    print(f'Checkpoint: {noisy.utils.rel_path(checkpoint.resolve())}')
    print(f'Iteration: {ctx.iteration}')
    print(f'Wandb ID: {ctx.wandb_run_id}')
    print(f'Loss EMA: {ctx.loss_ema}')
    print(f'Parameters: {round(total_params / 1e6, 2)} M')
    print(f'Checkpoint size: {round(cp_size / 1e6, 2)} MB')
    print(f'Dataset path: {cfg.data.path}')
    print(f'Dataset cache: {ds_cache if ds_cache.exists() else "(not found)"}')
    # Do not use len(ds) here as that would load the cache file, potentially
    # creating it.
    print(f'Images found: {len(ds_files)}')


@cli.command('slurm')
@click.option('--checkpoint', '-c', type=Path, default=None,
              help='Path to the checkpoint to be loaded.')
@click.option('--template', type=Path, default=Path('viking-script.sh.jinja'))
@click.option('--logfile', type=Path, default=None)
@click.option('--time', type=str, required=True)
@click.option('--email', type=str, default=None)
@click.option('--out', type=Path, default=None,
              help='The path to which the generated slurm script should be '
              'written to')
def slurm(checkpoint: Optional[Path], template: Path, logfile: Path, time: str,
          email: Optional[str], out: Optional[Path]) -> None:
    '''Submits a batch job to Slurm. For compute clusters.'''
    # Checks and preparations
    checkpoint = _ensure_cp(checkpoint, rel=False)
    if not template.exists():
        raise FileNotFoundError(f'Script does not exist: {template}')
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint}')
    if logfile is None:
        logfile = checkpoint.parent / 'viking.log'
    logfile = noisy.utils.rel_path(logfile.resolve())
    if out is None:
        out = checkpoint.parent / 'viking-script.sh'
    logfile.parent.mkdir(parents=True, exist_ok=True)
    # Prepare the script
    loader = jinja2.FileSystemLoader(searchpath='./')
    env = jinja2.Environment(loader=loader)
    templ = env.get_template(str(template))
    script = templ.render(
        email=email,
        logfile=logfile,
        time=time,
        noisy_dir=os.getcwd(),
        checkpoint=checkpoint
    )
    with open(out, 'w') as fh:
        fh.write(script)
    logger.info(f'Generated Slurm script written to {out}')


def _ensure_cp(checkpoint: Optional[Path], rel: bool = True) -> Path:
    if checkpoint is None:
        checkpoint = noisy.workdir.LATEST_PROJ_SL / noisy.workdir.LATEST_CP_SL
    if rel:
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
