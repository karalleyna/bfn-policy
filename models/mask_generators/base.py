import abc
from typing import Optional, Tuple

import torch
from torch import nn


class BaseMaskGenerator(nn.Module, abc.ABC):
    """Abstract base class for conditioning mask generators."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_dummy_buffer", torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self._dummy_buffer.device

    @abc.abstractmethod
    @torch.no_grad()
    def forward(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return self.forward(shape, generator)
