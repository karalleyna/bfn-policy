import abc
from typing import Any, Dict

import torch.nn as nn


class BaseNormalizer(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for data normalizers.

    This class provides a standardized interface for normalizing and un-normalizing
    data, which is a crucial preprocessing step in many machine learning pipelines.
    By defining a common structure, it allows for different normalization
    strategies to be used interchangeably.
    """

    def __init__(self):
        """
        Initializes the BaseNormalizer.
        """
        super().__init__()

    @abc.abstractmethod
    def normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes the input data.

        This method should be implemented by subclasses to apply a specific
        normalization transformation (e.g., min-max scaling, z-score).

        Args:
            data: A dictionary where keys are modality names and values are the
                  data to be normalized (e.g., PyTorch tensors).

        Returns:
            A dictionary with the same structure as the input, containing the
            normalized data.
        """
        raise NotImplementedError("Subclasses must implement the normalize method.")

    @abc.abstractmethod
    def unnormalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reverses the normalization transformation.

        This method is essential for converting the model's output, which is in
        the normalized space, back to the original data space.

        Args:
            data: A dictionary of normalized data.

        Returns:
            A dictionary with the same structure as the input, containing the
            un-normalized data.
        """
        raise NotImplementedError("Subclasses must implement the unnormalize method.")

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Defines the forward pass for the normalizer.

        This allows the normalizer to be used as a standard `nn.Module` within a
        larger network structure (e.g., `nn.Sequential`). By default, it calls
        the `normalize` method.

        Args:
            data: The input data to be normalized.

        Returns:
            The normalized data.
        """
        return self.normalize(data)
