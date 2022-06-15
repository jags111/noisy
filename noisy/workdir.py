from typing import Union, Tuple
from dataclasses import asdict
from logging import getLogger
from pathlib import Path
import shutil
import yaml
import torch
from torch.optim import AdamW

from .utils import AttrDict
from .models import Model
from .training import TrainingContext


logger = getLogger('noisy.workdir')
LATEST_PROJ_SL = Path('.latest')
LATEST_CP_SL = Path('.latest')


def init(wd: Path, cfg: Union[AttrDict, Path], force: bool = False) -> Path:
    # Load the config
    if isinstance(cfg, Path):
        cfg = AttrDict.from_yaml(cfg)
    # Remove the previous workdir if the force flag is set
    if force:
        logger.info('Removing existing directory: {workdir}')
        shutil.rmtree(str(wd))
    wd.mkdir(parents=True)
    # Create the initial checkpoint
    cp = wd / 'initial/'
    model = Model(cfg)
    optim = AdamW(model.parameters(), **cfg.optim)
    ctx = TrainingContext()
    save_checkpoint(cp, model, optim, cfg, ctx)
    return cp


def save_checkpoint(cp: Path, model: Model, optim: AdamW, cfg: AttrDict,
                    ctx: TrainingContext) -> None:
    cp.mkdir(exist_ok=True)
    save_cfg(cp, cfg)
    save_model(cp, model)
    save_optim(cp, optim)
    save_ctx(cp, ctx)


def load_checkpoint(cp: Path, device: torch.device,
                    ) -> Tuple[AttrDict, Model, AdamW, TrainingContext]:
    cfg = load_cfg(cp)
    model = load_model(cp, cfg)
    model.to(device)
    optim = load_opim(cp, cfg, model)
    ctx = load_ctx(cp)
    return cfg, model, optim, ctx


def save_cfg(cp: Path, cfg: AttrDict) -> None:
    cfg.to_yaml(cp / 'cfg.yaml')


def load_cfg(cp: Path) -> AttrDict:
    return AttrDict.from_yaml(cp / 'cfg.yaml')


def save_model(cp: Path, model: Model) -> None:
    state = model.state_dict()
    torch.save(state, cp / 'model.state.pt')


def load_model(cp: Path, cfg: AttrDict) -> Model:
    state = torch.load(cp / 'model.state.pt')  # type: ignore
    model = Model(cfg)
    model.load_state_dict(state)
    return model


def save_optim(cp: Path, opt: AdamW) -> None:
    state = opt.state_dict()
    torch.save(state, cp / 'opt.state.pt')


def load_opim(cp: Path, cfg: AttrDict, model: Model) -> AdamW:
    state = torch.load(cp / 'opt.state.pt')  # type: ignore
    optim = AdamW(model.parameters(), **cfg.optim)
    optim.load_state_dict(state)
    return optim


def save_ctx(cp: Path, ctx: TrainingContext) -> None:
    with open(cp / 'ctx.yaml', 'w') as fh:
        yaml.dump(asdict(ctx), fh)


def load_ctx(cp: Path) -> TrainingContext:
    with open(cp / 'ctx.yaml') as fh:
        d = yaml.safe_load(fh)
    return TrainingContext(**d)
