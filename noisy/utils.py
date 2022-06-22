'''Various utility functions and classes.'''
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import os
from logging import getLogger
from pathlib import Path
import torch
from torch import Tensor
import torchvision.utils as vutils  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import yaml


logger = getLogger('noisy.utils')


class AttrDict(Dict[str, Any]):
    '''A dictionary with syntax similar to that of JavaScript objects. I.e.
    instead of d['my_key'], we can simply say d.my_key.'''

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            if key not in self:
                raise e
            return self[key]

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AttrDict:
        return AttrDict(**{k: cls._from_obj(v) for k, v in d.items()})

    @classmethod
    def _from_obj(cls, o: Any) -> Any:
        if isinstance(o, dict):
            return cls.from_dict(o)
        if isinstance(o, list):
            return [cls._from_obj(x) for x in o]
        return o

    def to_dict(self) -> Dict[str, Any]:

        def _to_obj(x: Any) -> Any:
            if isinstance(x, AttrDict):
                return {k: _to_obj(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_to_obj(i) for i in x]
            return x

        obj = _to_obj(self)
        assert isinstance(obj, dict)
        return obj

    @classmethod
    def from_yaml(cls, path: Path) -> AttrDict:
        with open(path) as fh:
            d = yaml.safe_load(fh)
        return cls.from_dict(d)

    def to_yaml(self, path: Path) -> None:
        with open(path, 'w') as fh:
            yaml.dump(self.to_dict(), fh)


def make_symlink(sl: Path, cp: Path) -> None:
    if sl.exists():
        sl.unlink()
    sl.symlink_to(rel_path(cp.absolute(), sl.absolute().parent))


def rel_path(path: Path, root: Optional[Path] = None) -> Path:
    if root is None:
        root = Path(os.getcwd())
    return Path(os.path.relpath(path, root))


def show(img: Tensor, *, clip: bool = True, out: Optional[Path] = None
         ) -> None:
    '''Plots the given image.'''
    if img.dim() == 4 and img.size(0) != 1:
        return show_grid(img)
    if clip:
        img = torch.clip(img, 0., 1.)
    if isinstance(img, Tensor):
        img = (img
               .cpu()
               .numpy()
               .squeeze()
               .transpose((1, 2, 0)))  # C, H, W -> H, W, C
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    if out is None:
        plt.show()
    else:
        plt.savefig(out)


def show_grid(imgs: Tensor, *, clip: bool = True, out: Optional[Path] = None,
             figsize: Tuple[int, int] = (12, 12)) -> None:
    '''Plots the given images in a grid.'''
    imgs = imgs.detach().cpu()
    if clip:
        imgs = torch.clip(imgs, 0., 1.)
    grid = vutils.make_grid(imgs, padding=1, value_range=(0, 1))
    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.axis('off')
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.tight_layout()
    if out is None:
        plt.show()
    else:
        plt.savefig(out)


def random_name(prefix: str = 'run') -> str:
    counter_file = Path('.name_counter')
    if counter_file.exists():
        with open(counter_file, 'r') as fh:
            number = int(fh.read())
    else:
        number = 0
    with open(counter_file, 'w') as fh:
        fh.write(str(number + 1))
    return prefix + '-' + str(number).zfill(4)


def ema(x: float, acc: Optional[float], alpha: float = 0.95) -> float:
    '''Exponentially Moving Average.'''
    if acc is None:
        return x
    return alpha * acc + (1 - alpha) * x
