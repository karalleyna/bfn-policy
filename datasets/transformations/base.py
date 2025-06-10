import abc
from typing import Any, Callable, Dict

import numpy as np

from datasets.utils.replay_buffer import ReplayBuffer
from models.normalizers.base import BaseNormalizer


class BaseTransform(Callable):
    """Abstract base class for data transformations."""

    @abc.abstractmethod
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Applies the transformation to a raw data sample.

        Args:
            sample: A dictionary of numpy arrays from the replay buffer.

        Returns:
            A dictionary of processed data ready for model consumption.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_normalizer(self, replay_buffer: ReplayBuffer) -> "BaseNormalizer":
        """
        Creates a normalizer fitted to the entire dataset.

        Args:
            replay_buffer: The replay buffer containing the full dataset.

        Returns:
            A fitted normalizer object.
        """
        raise NotImplementedError
