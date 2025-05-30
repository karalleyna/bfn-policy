from abc import ABC, abstractmethod
from typing import Dict, Union

import torch
from torch.utils.data import Dataset as TorchDataset

from utils.normalizer import LinearNormalizer


class BaseDataset(TorchDataset, ABC):
    """
    Abstract base class for datasets used in lowdim and image-based policy learning.
    Subclasses must implement all abstract methods.
    """

    @abstractmethod
    def get_validation_dataset(self) -> "BaseDataset":
        """
        Returns a dataset object representing the validation set.
        """
        pass

    @abstractmethod
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        Returns a fitted normalizer for the dataset.
        """
        pass

    @abstractmethod
    def get_all_actions(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: All actions in the dataset as a (N, Da) tensor.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Returns:
            Dict with keys:
                - 'obs': torch.Tensor or Dict[str, torch.Tensor], shape (T, Do) or {"image": ..., "agent_pos": ...}
                - 'action': torch.Tensor, shape (T, Da)
        """
        pass
