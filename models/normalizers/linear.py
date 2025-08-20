from typing import Dict

import torch

from models.normalizers.base import BaseNormalizer


class LinearNormalizer(BaseNormalizer):
    """
    Normalizes data to the [-1, 1] range using min-max scaling.

    This normalizer is particularly useful for inputs to models with activation
    functions that are sensitive to the scale of the data, such as tanh.
    """

    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        """
        Initializes the LinearNormalizer by computing the min and max
        values from the provided data.

        Args:
            data_dict: A dictionary of tensors, where each tensor represents
                       the entire dataset for a specific modality.
        """
        super().__init__()
        for key, data in data_dict.items():
            # Register min and max as buffers to ensure they are moved to the
            # correct device and saved with the model's state_dict.
            self.register_buffer(f"min_{key}", data.min())
            self.register_buffer(f"max_{key}", data.max())

    def normalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Applies min-max scaling to normalize data to the [-1, 1] range.
        """
        normalized_data = {}
        for key, value in data.items():
            min_val = getattr(self, f"min_{key}")
            max_val = getattr(self, f"max_{key}")
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized_data[key] = 2 * (value - min_val) / (max_val - min_val) - 1
        return normalized_data

    def unnormalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Reverses the min-max scaling.
        """
        unnormalized_data = {}
        for key, value in data.items():
            min_val = getattr(self, f"min_{key}")
            max_val = getattr(self, f"max_{key}")
            # Formula: (x_norm + 1) / 2 * (max - min) + min
            unnormalized_data[key] = (value + 1) / 2 * (max_val - min_val) + min_val
        return unnormalized_data
