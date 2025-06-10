import abc

import torch
from torch import nn


class BaseEmbedding(nn.Module, abc.ABC):
    """
    Abstract base class for all embedding modules.

    This class defines the standard interface for modules that transform an
    input tensor into a fixed-size embedding vector. All concrete embedding
    implementations should inherit from this class.
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        """Returns the dimensionality of the embedding output."""
        return self._output_dim

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes an input tensor and returns its embedding.

        Args:
            x: The input tensor. Shape will vary depending on the embedding type.

        Returns:
            A tensor representing the embedding. Shape: (..., output_dim).
        """
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Provides a standard callable interface."""
        return self.forward(x)
