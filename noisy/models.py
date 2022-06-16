from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .utils import AttrDict


@dataclass
class Cursor:
    x: int = -1
    mult: int = 1

    def __call__(self, new_x: Optional[int] = None) -> int:
        if new_x is not None:
            self.x = new_x
        return self.x * self.mult


@dataclass(unsafe_hash=True)
class AssumeShape(nn.Module):
    assumed_shape: Tuple[int, int, int]

    def __post_init__(self) -> None:
        super().__init__()

    def forward(self, t: Tensor) -> Tensor:
        _, *s = t.shape
        if tuple(s) != self.assumed_shape:
            raise ValueError(f'Assumed shape {self.assumed_shape}, got: {s}')
        return t


def timestep_embedding(x: Tensor, t: Union[float, Tensor], n: int) -> Tensor:
    b, _, h, w = x.shape
    if isinstance(t, Tensor):
        b_, = t.shape
        assert b_ == b
        assert 0. <= t.min() and t.max() <= 1., t
        t = t.reshape(-1, 1, 1, 1)
    else:
        assert 0. <= t <= 1.
    embs = []
    for i in range(n):
        freq = 0.5 * torch.pi * t * 2 ** i
        emb = torch.ones((b, 1, h, w), device=x.device) * freq
        emb = torch.sin(emb)
        embs.append(emb)
    return torch.cat([*embs, x], dim=1)


class Residual(nn.Module):

    def __init__(self, *modules: nn.Module, gain: float = 1.) -> None:
        super().__init__()
        self.mods = nn.Sequential(*modules)
        self.gain = gain

    def forward(self, x: Tensor) -> Tensor:
        delta = self.mods(x)
        if self.gain != 1.:
            x = x * self.gain
        return delta + x


class AttentionBlock(nn.Module):
    '''Adapted from: https://github.com/pesser/pytorch_diffusion'''

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.InstanceNorm2d(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        # (1) Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h * w) # b,c,hw
        # b,hw,hw
        w_ = torch.bmm(q, k)
        w_ = w_ * c ** (-0.5)
        w_ = F.softmax(w_, dim=2)
        # (2) Attend to values
        v = v.reshape(b, c, h * w)
        # b,hw,hw (first hw of k, second of q)
        w_ = w_.permute(0, 2, 1)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class Model(nn.Module):

    def __init__(self, cfg: AttrDict) -> None:
        super().__init__()
        self.cfg = cfg
        c = Cursor(mult=self.cfg.arch.width_mult)
        gain = 1 / 2**0.5
        tsc = self.cfg.arch.timestep_channels
        self.main = nn.Sequential(
            # 64
            self.block(self.cfg.img.channels + tsc, c(2), gain=gain),
            nn.Conv2d(c(), c(), 3, 2, 1),
            self.nonlin(),
            # 32
            Residual(
                self.block(c(), c(4), gain=gain),
                nn.Conv2d(c(), c(), 3, 2, 1),
                self.nonlin(),
                # 16
                Residual(
                    self.block(c(), c(8), gain=gain),
                    AttentionBlock(c()),
                    nn.Conv2d(c(), c(), 3, 2, 1),
                    self.nonlin(),
                    # 8
                    Residual(
                        self.block(c(), c(16), gain=gain),
                        self.block(c(), c(), gain=gain),
                        AttentionBlock(c()),
                        nn.Conv2d(c(), c(8), 1, 1, 0),
                    ),
                    nn.ConvTranspose2d(c(), c(), 2, 2, 0),
                    # 16
                    self.nonlin(),
                    self.block(c(), c(), gain=gain),
                    AttentionBlock(c()),
                    nn.Conv2d(c(), c(4), 1, 1, 0),
                ),
                nn.ConvTranspose2d(c(), c(), 2, 2, 0),
                # 32
                self.nonlin(),
                self.block(c(), c(2), gain=gain),
                nn.Conv2d(c(), c(), 1, 1, 0),
            ),
            nn.ConvTranspose2d(c(), c(), 2, 2, 0),
            # 64
            self.nonlin(),
            self.block(c(), c(), gain=gain),
            nn.Conv2d(c(), self.cfg.img.channels, 1, 1, 0),
        )

    def nonlin(self) -> nn.Module:
        return nn.ReLU(inplace=True)

    def forward(self, x: Tensor, t: Union[float, Tensor]) -> Tensor:
        xt = timestep_embedding(x, t, self.cfg.arch.timestep_channels)
        delta = self.main(xt)
        # Scale the delta to have mean 0 and std 1/T
        delta = (delta - delta.mean()) / (delta.std() * self.cfg.T)
        return delta + x

    def block(self, in_c: int, c: int, gain: float = 1.) -> nn.Module:
        return nn.Sequential(
            nn.Identity() if in_c == c else nn.Conv2d(in_c, c, 1, 1, 0),
            Residual(
                nn.Conv2d(c, c, 3, 1, 1),
                self.nonlin(),
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(c),
                self.nonlin(),
                nn.Conv2d(c, c, 1, 1, 0),
                gain=gain,
            ),
        )

    def noise(self, n: int) -> Tensor:
        size = n, self.cfg.img.channels, self.cfg.img.size, self.cfg.img.size
        return torch.randn(size=size)
