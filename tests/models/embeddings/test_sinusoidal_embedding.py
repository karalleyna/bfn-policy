import math

import pytest
import torch
from torch import nn

# Assuming the classes from your canvas are in a file named `my_embeddings.py`
from models.embeddings.base import BaseEmbedding
from models.embeddings.mlp import MLPEmbedding
from models.embeddings.sinusoidal import SinusoidalTimestepEmbedding

# =========================== Test Fixtures (Reusable Setups) ===========================


@pytest.fixture
def device() -> torch.device:
    """Provides a device to run tests on (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================== Unit Test Classes ===========================


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


class TestMLPEmbedding:
    """A suite of unit tests for the MLPEmbedding class."""

    @pytest.fixture
    def mlp_config(self) -> dict:
        """Provides a standard configuration for the MLP embedder."""
        return {"input_dim": 10, "hidden_dims": (256, 128), "output_dim": 64}

    def test_initialization_and_properties(self, mlp_config):
        """Tests that the MLP initializes correctly and properties are set."""
        embedder = MLPEmbedding(**mlp_config)
        assert isinstance(embedder, BaseEmbedding)
        assert embedder.output_dim == mlp_config["output_dim"]
        # Check if the network has the correct number of layers (Linear + ReLU)
        # 2 hidden layers means 3 Linear layers. (in, h1), (h1, h2), (h2, out)
        # 2 ReLU layers. Total = 5 layers in Sequential (final ReLU is removed).
        assert len(embedder.network) == (len(mlp_config["hidden_dims"]) * 2)

    def test_forward_pass_shape_and_dtype(self, mlp_config, device):
        """Tests the output shape and dtype of the forward pass."""
        batch_size = 4
        embedder = MLPEmbedding(**mlp_config).to(device)
        input_tensor = torch.randn(batch_size, mlp_config["input_dim"]).to(device)

        embedding = embedder(input_tensor)

        assert embedding.shape == (batch_size, mlp_config["output_dim"])
        assert embedding.dtype == torch.float32
        assert embedding.device == device

    def test_network_structure(self, mlp_config):
        """Verifies the internal structure of the generated MLP."""
        embedder = MLPEmbedding(**mlp_config)

        # Check dimensions of linear layers
        assert isinstance(embedder.network[0], nn.Linear)
        assert embedder.network[0].in_features == mlp_config["input_dim"]
        assert embedder.network[0].out_features == mlp_config["hidden_dims"][0]

        assert isinstance(embedder.network[2], nn.Linear)
        assert embedder.network[2].in_features == mlp_config["hidden_dims"][0]
        assert embedder.network[2].out_features == mlp_config["hidden_dims"][1]

        # The last layer should be linear without a final ReLU
        assert isinstance(embedder.network[-1], nn.Linear)
        assert embedder.network[-1].in_features == mlp_config["hidden_dims"][-1]
        assert embedder.network[-1].out_features == mlp_config["output_dim"]
