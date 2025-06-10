import math

import torch

from models.embeddings.base import BaseEmbedding


class SinusoidalTimestepEmbedding(BaseEmbedding):
    """
    An embedding module for diffusion timesteps using sinusoidal position encodings.

    This module takes a 1D tensor of timesteps and maps each timestep to a
    high-dimensional vector using a combination of sine and cosine functions
    of varying frequencies. This technique was popularized by the Transformer
    architecture and is a standard way to encode timestep information in
    diffusion models.
    """

    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: The dimensionality of the output embedding.
        """
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"Embedding dimension must be divisible by 2, but got {embedding_dim}."
            )
        super().__init__(output_dim=embedding_dim)

        # Pre-calculate the frequency bands (the 'emb' tensor from the original code)
        # This is more efficient than recalculating it on every forward pass.
        half_dim = embedding_dim // 2

        # Formula from "Attention Is All You Need"
        exponent = -math.log(10000.0) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32) * exponent)

        # Register frequencies as a buffer. This makes it part of the module's
        # state, and it will be moved to the correct device with .to(), but it
        # is not considered a trainable parameter.
        self.register_buffer("frequencies", frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embeds the input timesteps.

        Args:
            x: A 1D tensor of timesteps. Shape: (B,).

        Returns:
            The embedding tensor. Shape: (B, embedding_dim).
        """
        if x.ndim != 1:
            raise ValueError(
                f"Input tensor must be 1D (a batch of timesteps), but got shape {x.shape}."
            )

        # Project the timesteps onto the frequency bands
        # Shape: (B, 1) * (1, half_dim) -> (B, half_dim)
        projections = x.unsqueeze(1).float() * self.frequencies.unsqueeze(0)

        # Concatenate the sine and cosine of the projections
        embeddings = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)

        return embeddings
