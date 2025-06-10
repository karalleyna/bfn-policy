import abc
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import zarr

# Define a flexible type for data sources
DataSource = Union[torch.Tensor, np.ndarray, zarr.Array, Dict[str, Any]]


class BaseNormalizer(nn.Module, abc.ABC):
    """Abstract base class for all normalizer implementations."""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def fit(self, data: DataSource, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def normalize(self, data: DataSource) -> DataSource:
        raise NotImplementedError

    @abc.abstractmethod
    def unnormalize(self, data: DataSource) -> DataSource:
        raise NotImplementedError
