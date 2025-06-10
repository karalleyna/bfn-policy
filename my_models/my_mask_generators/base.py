import abc
from typing import Optional, Tuple

import torch
from torch import nn


class BaseMaskGenerator(nn.Module, abc.ABC):
    """
    Abstract base class for conditioning mask generators.

    All mask generators should inherit from this class and implement the
    `forward` method. Inheriting from `nn.Module` allows PyTorch to manage
    the device placement of any internal tensors.
    """

    def __init__(self):
        super().__init__()
        # --- FIX IS HERE ---
        # Register a dummy buffer to make the module device-aware.
        # This buffer will be moved to the correct device when .to(device)
        # is called on the module, allowing self.device to work correctly.
        self.register_buffer("_dummy_buffer", torch.empty(0))

    @property
    def device(self) -> torch.device:
        """A convenience property to get the device of the module."""
        return self._dummy_buffer.device

    @abc.abstractmethod
    @torch.no_grad()
    def forward(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generates a boolean conditioning mask.

        Args:
            shape: The desired shape of the mask, typically (B, T, D), where
                   B=batch size, T=horizon, D=feature dimension.
            generator: A PyTorch random number generator for reproducibility.

        Returns:
            A boolean tensor of the given shape. `True` indicates a
            conditioned (known) value, while `False` indicates an unconditioned
            (to be predicted) value.
        """
        raise NotImplementedError

    def __call__(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Provides a standard callable interface."""
        return self.forward(shape, generator)
