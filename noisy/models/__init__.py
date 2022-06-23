'''The model architecture. A combination of Convolutions and attention blocks,
with multiple residual connections.'''
from __future__ import annotations
from typing import Union, Callable, Type, Dict, TypeVar
from logging import getLogger
import torch

from noisy.utils import AttrDict
from .base import Model


__all__ = 'Model', 'get_model_class', 'model_ema', 'model_ema_init'


logger = getLogger('noisy.models')


MODELS: Dict[str, Type[Model]] = dict()


def register_model(name: str) -> Callable[[Type[Model]], Type[Model]]:
    def wrapper(cls: Type[Model]) -> Type[Model]:
        MODELS[name] = cls
        return cls
    return wrapper


def get_model_class(name_or_cfg: Union[str, AttrDict]) -> Type[Model]:
    from . import lara  # noqa:  F401
    from . import maria  # noqa:  F401
    if isinstance(name_or_cfg, AttrDict):
        name = name_or_cfg.arch.model
    else:
        name = name_or_cfg
    if name not in MODELS:
        raise ValueError(f'Unknown model: {name}, expected one of '
                         f'{list(MODELS.keys())}')
    return MODELS[name]


ModelT = TypeVar('ModelT', bound=Model)


def model_ema(ema: ModelT, model: ModelT, alpha: float, steps: int
              ) -> ModelT:
    with torch.no_grad():
        a = alpha ** steps
        for ema_p, model_p in zip(ema.parameters(), model.parameters()):
            other_data = model_p.data.clone().detach().to(ema_p.device)
            ema_p.data = a * ema_p + (1 - a) * other_data
    return ema


def model_ema_init(model: ModelT) -> ModelT:
    ema = model.__class__(model.cfg)
    for ema_p, model_p in zip(ema.parameters(), model.parameters()):
        ema_p.data = model_p.data.clone().detach().to(ema_p.device)
    return ema
