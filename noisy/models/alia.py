from typing import Union
import torch
from torch import Tensor
import torch.nn as nn

from noisy.utils import AttrDict
from .common import (Cursor, Residual, AttentionBlock, ConvUpsample,
                     ConvDownsample, timestep_embedding)
from .base import Model
from . import register_model


@register_model('Alia')
class Alia(Model):
    '''The diffusion model.'''

    def __init__(self, cfg: AttrDict) -> None:
        super().__init__(cfg)
        c = Cursor(mult=self.cfg.arch.width_mult)
        gain = 1.
        tsc = self.cfg.arch.timestep_channels
        self.main = nn.Sequential(
            # 16
            self.block(self.cfg.img.channels + tsc, c(1), gain=gain),
            self.block(c(), c(), gain=gain),
            self.block(c(), c(), gain=gain),
            self.downsample(c()),
            self.nonlin(),
            # 8
            Residual(
                self.block(c(), c(2), gain=gain),
                self.block(c(), c(), gain=gain),
                self.block(c(), c(), gain=gain),
                self.downsample(c()),
                # 4
                Residual(
                    self.block(c(), c(4), gain=gain),
                    self.block(c(), c(), gain=gain),
                    self.block(c(), c(), gain=gain),
                    AttentionBlock(c()),
                    self.downsample(c()),
                    # 2
                    Residual(
                        self.block(c(), c(8), gain=gain),
                        self.block(c(), c(), gain=gain),
                        AttentionBlock(c()),
                        self.block(c(), c(), gain=gain),
                        AttentionBlock(c()),
                        self.downsample(c()),
                        # 1
                        Residual(
                            self.block(c(), c(16), gain=gain),
                            AttentionBlock(c()),
                            self.block(c(), c(), gain=gain),
                            AttentionBlock(c()),
                            self.block(c(), c(), gain=gain),
                            AttentionBlock(c()),
                            self.block(c(), c(), gain=gain),
                            AttentionBlock(c()),
                            nn.Conv2d(c(), c(8), 1, 1, 0),
                        ),
                        self.upsample(c()),
                        # 2
                        self.block(c(), c(), gain=gain),
                        self.block(c(), c(), gain=gain),
                        AttentionBlock(c()),
                        self.block(c(), c(), gain=gain),
                        AttentionBlock(c()),
                        nn.Conv2d(c(), c(4), 1, 1, 0),
                    ),
                    self.upsample(c()),
                    # 4
                    self.block(c(), c(), gain=gain),
                    self.block(c(), c(), gain=gain),
                    self.block(c(), c(), gain=gain),
                    AttentionBlock(c()),
                    nn.Conv2d(c(), c(2), 1, 1, 0),
                ),
                self.upsample(c()),
                # 8
                self.block(c(), c(), gain=gain),
                self.block(c(), c(), gain=gain),
                self.block(c(), c(), gain=gain),
                nn.Conv2d(c(), c(1), 1, 1, 0),
            ),
            self.upsample(c()),
            # 16
            self.block(c(), c(), gain=gain),
            self.block(c(), c(), gain=gain),
            self.block(c(), c(), gain=gain),
            nn.Conv2d(c(), self.cfg.img.channels, 1, 1, 0),
        )

    def nonlin(self) -> nn.Module:
        return nn.SiLU(inplace=True)

    def downsample(self, c: int) -> nn.Module:
        return ConvDownsample(c)

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
