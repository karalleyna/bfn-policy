"""
Defines the absolute base interface for all policies.
"""

import abc
from typing import Dict, Optional

import torch
import torch.nn as nn


class AbstractPolicy(nn.Module, abc.ABC):
    """
    Top-level abstract base class for all policies.

    This class defines the essential contract for any policy within the framework.
    It mandates the implementation of `predict_action` for inference and
    `compute_loss` for training. It also manages a data normalizer, which
    is a common requirement for most policies.
    """

    def __init__(self):
        super().__init__()
        # The normalizer is essential for preprocessing data before it's used
        # by the model. It's expected to be set externally.
        self._normalizer: Optional[Dict[str, torch.Tensor]] = None

    @property
    def normalizer(self) -> Dict[str, torch.Tensor]:
        """Provides access to the data normalizer."""
        if self._normalizer is None:
            raise RuntimeError("Normalizer has not been set!")
        return self._normalizer

    def set_normalizer(self, normalizer: Dict[str, torch.Tensor]):
        """Sets the data normalizer for the policy."""
        self._normalizer = normalizer

    @abc.abstractmethod
    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Generates an action based on the provided observation.

        Args:
            obs_dict: A dictionary of observations from the environment.

        Returns:
            A dictionary containing the predicted action.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the training loss for a batch of data.

        Args:
            batch: A dictionary containing a batch of training data, typically
                   including observations and ground-truth actions.

        Returns:
            A scalar tensor representing the computed loss.
        """
        raise NotImplementedError
