from typing import Optional
import torch
from torch import Tensor
from tqdm import tqdm  # type: ignore

from .models import Model
from .utils import AttrDict


def synthesize(model: Model, cfg: AttrDict, number: int,
               iterations: Optional[int] = None,
               std: Optional[float] = None,
               device: Optional[torch.device] = None,
               show_bar: bool = False) -> Tensor:
    '''Sample the model.'''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if iterations is None:
        iterations = int(cfg.T)
    if std is None:
        std = 1 / iterations
    model.to(device).eval()
    shape = (number, cfg.img.channels, cfg.img.size, cfg.img.size)
    img = torch.randn(shape, device=device)
    # Prepare the iterator over 0..iterations.
    it = range(iterations)
    if show_bar:
        it = tqdm(it)
    # Run the model.
    with torch.no_grad():
        for t in it:
            t_ = t / iterations
            img = img + model(img, t_, std=std)
    return img
