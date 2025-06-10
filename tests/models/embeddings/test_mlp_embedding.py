import pytest
import torch
from torch import nn

# Assuming the classes from your canvas are in a file named `my_embeddings.py`
from models.embeddings.base import BaseEmbedding
from models.embeddings.mlp import MLPEmbedding


@pytest.fixture
def device() -> torch.device:
    """Provides a device to run tests on (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # --- FIX IS HERE ---
        # The logic for calculating the number of layers was incorrect.
        # For N hidden layers, there are (N+1) Linear layers and N ReLU layers.
        # The implementation removes the final ReLU, so the total number of layers is:
        # (N+1) Linear layers + N ReLU layers = 2*N + 1
        num_hidden_layers = len(mlp_config["hidden_dims"])
        expected_num_layers = (num_hidden_layers * 2) + 1
        assert len(embedder.network) == expected_num_layers

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
