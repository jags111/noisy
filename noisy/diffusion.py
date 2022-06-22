from typing import Union, Iterator, Optional
from logging import getLogger
import torch
from torch import Tensor
from tqdm import tqdm

from .utils import AttrDict
from .models import Model


logger = getLogger('noisy.diffusion')


def _resolve_device(device: Optional[Union[str, torch.device]]
                    ) -> torch.device:
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, str):
        return torch.device(device)
    return device


def _resolve_img(img_or_n: Union[int, Tensor], cfg: AttrDict,
                 device: torch.device) -> Tensor:
    if isinstance(img_or_n, int):
        s = cfg.img.size
        x = torch.randn((img_or_n, cfg.img.channels, s, s), device=device)
    else:
        x = img_or_n
    return x


def get_beta_sched(cfg: AttrDict) -> Tensor:
    T = cfg.diffusion.T
    low = cfg.diffusion.beta_start
    high = cfg.diffusion.beta_stop
    sched = torch.arange(T) / T
    sched = low + sched * (high - low)
    assert isinstance(sched, Tensor)
    return sched


def get_alpha_sched(betas: Tensor) -> Tensor:
    return 1 - betas  # type: ignore


def get_alpha_cumprod_sched(betas: Tensor) -> Tensor:
    alphas = get_alpha_sched(betas)
    return torch.cumprod(alphas, dim=0)


def sample(model: Model, cfg: AttrDict, img_or_n: Union[Tensor, int] = 1,
           device: Optional[Union[str, torch.device]] = None,
           show_bar: bool = False) -> Tensor:
    device = _resolve_device(device)
    img = _resolve_img(img_or_n, cfg, device)
    *_, img = sample_iter(model, img, get_beta_sched(cfg), cfg.diffusion.T,
                          device, show_bar)
    return img


def sample_iter(model: Model, x: Tensor, beta_sched: Tensor, T: int,
                device: torch.device, show_bar: bool = False
                ) -> Iterator[Tensor]:
    model.eval().to(device)
    with torch.no_grad():
        alpha_cumprods = get_alpha_cumprod_sched(beta_sched)
        alphas = get_alpha_sched(beta_sched)
        it = reversed(range(T))
        if show_bar:
            it = tqdm(it, total=T)
        for t in it:
            a = alphas[t]
            ac = alpha_cumprods[t]
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            model_fac = (1 - a) / ((1 - ac).sqrt())
            noise_fac = (1 - a).sqrt()
            delta = model(x, t)
            x = 1 / a.sqrt() * (x - model_fac * delta) + noise_fac * z
            yield x
    return x
