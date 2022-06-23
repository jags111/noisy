from typing import Union
from abc import ABC, abstractmethod
from torch import Tensor
import torch.nn as nn

from noisy.utils import AttrDict


class Model(nn.Module, ABC):

    def __init__(self, cfg: AttrDict) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, x: Tensor, t: Union[int, Tensor]) -> Tensor:
        raise NotImplementedError
