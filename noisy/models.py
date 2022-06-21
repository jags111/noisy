'''The model architecture. A combination of Convolutions and attention blocks,
with multiple residual connections.'''
from __future__ import annotations
from typing import Optional, Tuple, Union
from dataclasses import dataclass
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


def timestep_embedding(ctx: ExecContext, n: int) -> ExecContext:
    '''Adds the timestep t as a collection of channels, 1, 2, ..., I, each with
    values roughly equal to sin(2 ** i * t). Somewhat similar to the embeddings
    common.y used in transformers.'''
    x, t = ctx.x, ctx.t
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
    ctx.x = torch.cat([*embs, x], dim=1)
    return ctx


class Residual(nn.Module):
    '''Similar to torch.nn.Sequential, only with a residual connection.'''

    def __init__(self, *modules: nn.Module, gain: float = 1.) -> None:
        super().__init__()
        self.mods = nn.Sequential(*modules)
        self.gain = gain

    def forward(self, ctx: ExecContext) -> ExecContext:
        x = ctx.x
        delta = self.mods(x)
        if self.gain != 1.:
            x = x * self.gain
        ctx.x = delta + x
        return ctx


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


class X(nn.Module):
    '''Only passes the `x` attribute of the execution context to the contained
    module, updating the execution context along the way.'''

    def __init__(self, mod: nn.Module) -> None:
        super().__init__()
        self.mod = mod

    def forward(self, ctx: ExecContext) -> ExecContext:
        ctx.x = self.mod(ctx.x)
        return ctx


@dataclass
class ExecContext:
    '''Container for the data that is propagated through the forward pass of
    the network. This saves us from implementng a stateful `forward` method
    that keeps track of the intermediate results. Instead, we can define some
    helper functions and classes (concretely the `Residual` module and the `X`
    module).'''
    x: Tensor
    t: Tensor


class Model(nn.Module):
    '''The diffusion model.'''

    def __init__(self, cfg: AttrDict) -> None:
        super().__init__()
        self.cfg = cfg
        c = Cursor(mult=self.cfg.arch.width_mult)
        gain = 1.
        tsc = self.cfg.arch.timestep_channels
        self.main = nn.Sequential(
            # 64
            X(self.block(self.cfg.img.channels + tsc, c(2), gain=gain)),
            X(nn.Conv2d(c(), c(), 3, 2, 1)),
            X(self.nonlin()),
            # 32
            Residual(
                X(self.block(c(), c(4), gain=gain)),
                X(nn.Conv2d(c(), c(), 3, 2, 1)),
                X(self.nonlin()),
                # 16
                Residual(
                    X(self.block(c(), c(8), gain=gain)),
                    X(AttentionBlock(c())),
                    X(nn.Conv2d(c(), c(), 3, 2, 1)),
                    X(self.nonlin()),
                    # 8
                    Residual(
                        X(self.block(c(), c(16), gain=gain)),
                        X(self.block(c(), c(), gain=gain)),
                        X(AttentionBlock(c())),
                        X(nn.Conv2d(c(), 1, 1, 0)),
                    ),
                    X(nn.ConvTranspose2d(c(), c(), 2, 2, 0)),
                    # 16
                    X(self.nonlin()),
                    X(self.block(c(), c(), gain=gain)),
                    X(AttentionBlock(c())),
                    X(nn.Conv2d(c(), c(4), 1, 1, 0)),
                ),
                X(nn.ConvTranspose2d(c(), c(), 2, 2, 0)),
                # 32
                X(self.nonlin()),
                X(self.block(c(), c(2), gain=gain)),
                X(nn.Conv2d(c(), c(), 1, 1, 0)),
            ),
            X(nn.ConvTranspose2d(c(), c(), 2, 2, 0)),
            # 64
            X(self.nonlin()),
            X(self.block(c(), c(), gain=gain)),
            X(nn.Conv2d(c(), self.cfg.img.channels, 1, 1, 0)),
        )

    def nonlin(self) -> nn.Module:
        return nn.ReLU(inplace=True)

    def forward(self, x: Tensor, t: Union[float, Tensor], std: float = 1.
                ) -> Tensor:
        if isinstance(t, float):
            t = torch.tensor([t] * x.size(0))
        ctx = ExecContext(x=x, t=t)
        ctx = timestep_embedding(ctx, self.cfg.arch.timestep_channels)
        delta = self.main(ctx)
        # Scale the delta to have mean 0 and std `std`
        delta = (delta - delta.mean()) / (delta.std() / std)
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
