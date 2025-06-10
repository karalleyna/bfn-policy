import math

import pytest
import torch

from models.embeddings.base import BaseEmbedding
from models.embeddings.sinusoidal import SinusoidalTimestepEmbedding


@pytest.fixture
def device() -> torch.device:
    """Provides a device to run tests on (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSinusoidalTimestepEmbedding:
    """A suite of unit tests for the SinusoidalTimestepEmbedding class."""

    def test_initialization_and_properties(self):
        """Tests that the module initializes correctly and properties are set."""
        embedding_dim = 128
        embedder = SinusoidalTimestepEmbedding(embedding_dim=embedding_dim)
        assert isinstance(embedder, BaseEmbedding)
        assert embedder.output_dim == embedding_dim
        assert hasattr(embedder, "frequencies")
        assert embedder.frequencies.shape == (embedding_dim // 2,)

    def test_initialization_fails_on_odd_dimension(self):
        """Tests that initialization raises a ValueError for an odd embedding dimension."""
        with pytest.raises(ValueError, match="must be divisible by 2"):
            SinusoidalTimestepEmbedding(embedding_dim=129)

    def test_forward_pass_shape_and_dtype(self, device):
        """Tests the output shape and dtype of the forward pass."""
        batch_size = 4
        embedding_dim = 64
        embedder = SinusoidalTimestepEmbedding(embedding_dim=embedding_dim).to(device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)

        embedding = embedder(timesteps)

        assert embedding.shape == (batch_size, embedding_dim)
        assert embedding.dtype == torch.float32
        assert embedding.device == device

    def test_forward_pass_correctness(self):
        """
        Tests the mathematical correctness of the embedding by comparing against
        a manually calculated value for a known input.
        """
        embedding_dim = 4
        embedder = SinusoidalTimestepEmbedding(embedding_dim=embedding_dim)

        # Test with a single known timestep, e.g., t=50
        t = torch.tensor([50.0])

        # Manual calculation based on the formula
        half_dim = embedding_dim // 2
        exponent = -math.log(10000.0) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32) * exponent)
        projections = t.unsqueeze(1) * frequencies.unsqueeze(0)
        expected_embedding = torch.cat(
            [torch.sin(projections), torch.cos(projections)], dim=-1
        )

        # Get the actual output
        actual_embedding = embedder(t)

        torch.testing.assert_close(actual_embedding, expected_embedding)

    def test_forward_pass_invalid_input_shape(self):
        """Tests that the forward pass raises a ValueError for incorrect input dimensions."""
        embedder = SinusoidalTimestepEmbedding(embedding_dim=128)

        # Input should be 1D, so a 2D tensor should fail
        bad_input = torch.randint(0, 100, (4, 1))
        with pytest.raises(ValueError, match="Input tensor must be 1D"):
            embedder(bad_input)
