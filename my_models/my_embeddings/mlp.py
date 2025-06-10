from typing import Tuple

import torch
from torch import nn

from my_models.my_embeddings.base import BaseEmbedding


class MLPEmbedding(BaseEmbedding):
    """
    A generic embedding module using a Multi-Layer Perceptron (MLP).

    This module can be used to create an embedding from any flat vector input
    by passing it through a series of linear layers and activation functions.
    """

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int):
        """
        Args:
            input_dim: The dimensionality of the input vector.
            hidden_dims: A tuple of integers specifying the size of each hidden layer.
            output_dim: The final dimensionality of the embedding.
        """
        super().__init__(output_dim=output_dim)

        all_dims = [input_dim] + list(hidden_dims) + [output_dim]

        layers = []
        for in_d, out_d in zip(all_dims[:-1], all_dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())

        # Remove the last ReLU to have a linear output layer
        self.network = nn.Sequential(*layers[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the MLP to create the embedding.

        Args:
            x: The input tensor. Shape: (B, input_dim).

        Returns:
            The embedding tensor. Shape: (B, output_dim).
        """
        return self.network(x)
