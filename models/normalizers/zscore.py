from typing import Dict

import torch

from models.normalizers.base import BaseNormalizer


class ZScoreNormalizer(BaseNormalizer):
    """
    Normalizes data to have zero mean and unit variance (z-score normalization).

    This is a standard technique that can help models converge faster and is
    less sensitive to outliers compared to min-max scaling.
    """

    def __init__(self, data_dict: Dict[str, torch.Tensor], epsilon: float = 1e-8):
        """
        Initializes the ZScoreNormalizer by computing the mean and standard
        deviation from the provided data.

        Args:
            data_dict: A dictionary of tensors for each modality.
            epsilon: A small value to add to the standard deviation to prevent
                     division by zero.
        """
        super().__init__()
        self.epsilon = epsilon
        for key, data in data_dict.items():
            self.register_buffer(f"mean_{key}", data.mean())
            self.register_buffer(f"std_{key}", data.std())

    def normalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Applies z-score normalization.
        """
        normalized_data = {}
        for key, value in data.items():
            mean = getattr(self, f"mean_{key}")
            std = getattr(self, f"std_{key}")
            normalized_data[key] = (value - mean) / (std + self.epsilon)
        return normalized_data

    def unnormalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Reverses the z-score normalization.
        """
        unnormalized_data = {}
        for key, value in data.items():
            mean = getattr(self, f"mean_{key}")
            std = getattr(self, f"std_{key}")
            unnormalized_data[key] = value * (std + self.epsilon) + mean
        return unnormalized_data
