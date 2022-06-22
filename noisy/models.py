'''The model architecture. A combination of Convolutions and attention blocks,
with multiple residual connections.'''
from __future__ import annotations
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .utils import AttrDict


@dataclass
class Cursor:
    '''A convenience class that helps tracking the current number of channels
    in the model.'''
    x: int = -1
    mult: int = 1

    def __call__(self, new_x: Optional[int] = None) -> int:
        if new_x is not None:
            self.x = new_x
        return self.x * self.mult


@dataclass(unsafe_hash=True)
class AssumeShape(nn.Module):
    '''A convenience class that helps to debug model architectures.'''
    assumed_shape: Tuple[int, int, int]

    def __post_init__(self) -> None:
        super().__init__()

    def forward(self, t: Tensor) -> Tensor:
        _, *s = t.shape
        if tuple(s) != self.assumed_shape:
            raise ValueError(f'Assumed shape {self.assumed_shape}, got: {s}')
        return t


def timestep_embedding(t: Tensor, dim: int, device: torch.device) -> Tensor:
    '''Adds the timestep t as a collection of channels, 1, 2, ..., I, each with
    values roughly equal to sin(2 ** i * t). Somewhat similar to the embeddings
    common.y used in transformers.'''
    assert dim % 2 == 0, f'dim should be even but is {dim}'
    span = math.log(10000) / (dim // 2 - 1)
    emb = torch.exp(torch.arange(dim // 2, device=device) * -span)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


class Residual(nn.Module):
    '''Similar to torch.nn.Sequential, only with a residual connection.'''

    def __init__(self, *modules: nn.Module, gain: float = 1.) -> None:
        super().__init__()
        self.mods = nn.Sequential(*modules)
        self.gain = gain

    def forward(self, x: Tensor) -> Tensor:
        delta = self.mods(x)
        if self.gain != 1.:
            x = x * self.gain
        x = delta + x
        return x


class AttentionBlock(nn.Module):
    '''Adapted from: https://github.com/pesser/pytorch_diffusion'''

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.InstanceNorm2d(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = torch.nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        # (1) Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)   # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        w_ = torch.bmm(q, k)  # b, hw, hw
        w_ = w_ / c ** 0.5
        w_ = F.softmax(w_, dim=2)
        # (2) Attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b, hw, hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c, hw (hw of q) h_[b, c, j]
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class ConvUpsample(nn.Module):

    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(c, c, 2, 2, 0)
        self.up = nn.Upsample(scale_factor=2.)

    def forward(self, x: Tensor) -> Tensor:
        up = self.up(x)
        dx = self.conv(x)
        x = up + dx
        assert isinstance(x, Tensor)
        return x


class Model(nn.Module):
    '''The diffusion model.'''

    def __init__(self, cfg: AttrDict) -> None:
        super().__init__()
        self.cfg = cfg
        c = Cursor(mult=self.cfg.arch.width_mult)
        gain = 1.
        tsc = self.cfg.arch.timestep_channels
        self.main = nn.Sequential(
            # 8
            self.block(self.cfg.img.channels + tsc, c(2), gain=gain),
            self.downsample(c(), c()),
            self.nonlin(),
            # 4
            Residual(
                self.block(c(), c(4), gain=gain),
                self.downsample(c(), c()),
                self.nonlin(),
                # 2
                Residual(
                    self.block(c(), c(8), gain=gain),
                    AttentionBlock(c()),
                    self.downsample(c(), c()),
                    self.nonlin(),
                    # 1
                    Residual(
                        self.block(c(), c(16), gain=gain),
                        self.block(c(), c(), gain=gain),
                        AttentionBlock(c()),
                        nn.Conv2d(c(), c(8), 1, 1, 0),
                    ),
                    self.upsample(c()),
                    # 2
                    self.nonlin(),
                    self.block(c(), c(), gain=gain),
                    AttentionBlock(c()),
                    nn.Conv2d(c(), c(4), 1, 1, 0),
                ),
                self.upsample(c()),
                # 4
                self.nonlin(),
                self.block(c(), c(2), gain=gain),
                nn.Conv2d(c(), c(), 1, 1, 0),
            ),
            self.upsample(c()),
            # 8
            self.nonlin(),
            self.block(c(), c(), gain=gain),
            nn.Conv2d(c(), self.cfg.img.channels, 1, 1, 0),
        )

    def nonlin(self) -> nn.Module:
        return nn.ReLU(inplace=True)

    def downsample(self, ch_in: int, ch_out: int) -> nn.Module:
        return nn.Conv2d(ch_in, ch_out, 3, 2, 1)

    def upsample(self, c: int) -> nn.Module:
        return ConvUpsample(c)

    def forward(self, x: Tensor, t: Union[int, Tensor]) -> Tensor:
        batch_size, _, h, w = x.shape
        if isinstance(t, int):
            t = torch.tensor([t] * batch_size, device=x.device)
        ts = timestep_embedding(t, self.cfg.arch.timestep_channels,
                                x.device)
        ts = ts.reshape([batch_size, -1, 1, 1]).expand(batch_size, -1, h, w)
        inputs = torch.cat([ts, x], dim=1)
        delta = self.main(inputs)
        assert isinstance(delta, Tensor)
        return delta

    def block(self, in_c: int, c: int, gain: float = 1.) -> nn.Module:
        return nn.Sequential(
            nn.Identity() if in_c == c  # type: ignore
            else nn.Conv2d(in_c, c, 1, 1, 0),
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
