import abc
from typing import Any, Dict

import torch
import torch.nn as nn


class Policy(nn.Module):
    """Top-level abstract base class for policies."""

    def __init__(self):
        super().__init__()
        self.normalizer = None

    def to(self, *args, **kwargs):
        # Move all submodules to the specified device/dtype
        for module in self.modules():
            if module is not self:
                module.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")  # Default

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32  # Default

    def set_normalizer(self, normalizer: Any):
        self.normalizer = normalizer

    @abc.abstractmethod
    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
