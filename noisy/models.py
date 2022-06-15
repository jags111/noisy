import torch
from torch import Tensor
import torch.nn as nn

from .utils import AttrDict


class Model(nn.Module):

    def __init__(self, cfg: AttrDict) -> None:
        super().__init__()
        self.cfg = cfg
        self.main = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1, 1),
        )

    def forward(self, img: Tensor, t: float) -> Tensor:
        assert 0 <= t <= 1.
        b, _, h, w = img.shape
        tt = torch.ones((b, 1, h, w), device=img.device) * t
        imgt = torch.cat((img, tt), dim=1)
        img = self.main(imgt) + img
        assert isinstance(img, Tensor)
        return img
