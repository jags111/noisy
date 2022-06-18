from typing import Optional, Iterator, Union
import torch
from torch import Tensor
from tqdm import tqdm  # type: ignore

from .models import Model
from .utils import AttrDict


def _syn_step(model: Model, t: float, img: Tensor, std: float) -> Tensor:
    with torch.no_grad():
        return img + model(img, t, std)


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, str):
        return torch.device(device)
    return device


def _resolve_img(img: Optional[Tensor], cfg: AttrDict, device: torch.device,
                 number: int) -> Tensor:
    shape = (number, cfg.img.channels, cfg.img.size, cfg.img.size)
    img_ = torch.randn(shape, device=device) if img is None else img
    assert img_.shape == (number, cfg.img.channels, cfg.img.size, cfg.img.size)
    return img_


def synthesize(model: Model, cfg: AttrDict, number: int,
               iterations: Optional[int] = None,
               std: Optional[float] = None,
               device: Optional[torch.device] = None,
               show_bar: bool = False,
               img: Optional[Tensor] = None) -> Tensor:
    '''Sample the model.'''
    it = synthesize_iter(model, cfg, number, yield_every=None,
                         iterations=iterations, std=std,
                         device=device, show_bar=show_bar, img=img)
    return next(it)


def synthesize_iter(model: Model, cfg: AttrDict, number: int,
                    yield_every: Optional[int] = 1,
                    iterations: Optional[int] = None,
                    std: Optional[float] = None,
                    device: Optional[torch.device] = None,
                    show_bar: bool = False,
                    img: Optional[Tensor] = None) -> Iterator[Tensor]:
    device_ = _resolve_device(device)
    model.to(device).eval()
    img_ = _resolve_img(img, cfg, device_, number)
    if iterations is None:
        iterations = int(cfg.T)
    if yield_every is None:
        yield_every = iterations
    if iterations % yield_every != 0:
        raise ValueError(f'iterations should be divisible by yield_every, but '
                         f'is {iterations} and {yield_every} respectively.')
    if std is None:
        std = 2 / iterations
    # Prepare the iterator over 0..iterations.
    it = range(iterations)
    if show_bar:
        it = tqdm(it)
    # Run the model.
    for t in it:
        if (t + 1) % yield_every == 0:
            yield img_
        img_ = _syn_step(model, t / iterations, img_, std)
