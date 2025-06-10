from typing import Any, Dict

import pytest
import torch

from models.conditional_unet1d import ConditionalResidualBlock1D, ConditionalUnet1D


@pytest.fixture
def device() -> torch.device:
    """Provides a device to run tests on (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestConditionalResidualBlock1D:
    """A suite of unit tests for the ConditionalResidualBlock1D class."""

    @pytest.fixture
    def resblock_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for the residual block."""
        return {
            "in_channels": 64,
            "out_channels": 128,
            "cond_dim": 256,
            "kernel_size": 3,
            "n_groups": 8,
        }

    def test_output_shape(self, resblock_config, device):
        """Tests that the block produces an output with the correct shape."""
        batch_size, seq_len = 4, 16
        block = ConditionalResidualBlock1D(**resblock_config).to(device)

        x = torch.randn(batch_size, resblock_config["in_channels"], seq_len).to(device)
        cond = torch.randn(batch_size, resblock_config["cond_dim"]).to(device)

        output = block(x, cond)

        expected_shape = (batch_size, resblock_config["out_channels"], seq_len)
        assert output.shape == expected_shape

    def test_film_conditioning_scale_and_bias(self, resblock_config, device):
        """Tests the FiLM logic with both scale and bias prediction."""
        block = ConditionalResidualBlock1D(
            **resblock_config, cond_predict_scale=True
        ).to(device)

        # Check that the FiLM projection layer has the correct output dimension (2 * out_channels)
        expected_film_dim = resblock_config["out_channels"] * 2
        assert block.film_projection[-1].out_features == expected_film_dim

    def test_film_conditioning_bias_only(self, resblock_config, device):
        """Tests the FiLM logic with only bias prediction."""
        block = ConditionalResidualBlock1D(
            **resblock_config, cond_predict_scale=False
        ).to(device)

        # Check that the FiLM projection layer has the correct output dimension (out_channels)
        expected_film_dim = resblock_config["out_channels"]
        assert block.film_projection[-1].out_features == expected_film_dim


class TestConditionalUnet1D:
    """A suite of unit tests for the ConditionalUnet1D model."""

    @pytest.fixture
    def unet_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for the U-Net model."""
        return {
            "input_dim": 10,
            "global_cond_dim": 64,
            "down_dims": [256, 512],
        }

    @pytest.fixture
    def sample_data(
        self, unet_config: Dict[str, Any], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Provides a sample batch of data for testing the U-Net."""
        batch_size, horizon = 4, 16
        return {
            "sample": torch.randn(batch_size, horizon, unet_config["input_dim"]).to(
                device
            ),
            "timestep": torch.randint(0, 100, (batch_size,), device=device),
            "global_cond": torch.randn(batch_size, unet_config["global_cond_dim"]).to(
                device
            ),
        }

    def test_forward_pass_output_shape(self, unet_config, sample_data, device):
        """
        Tests that a full forward pass runs without errors and produces an output
        tensor of the same shape as the input sample. This is the primary
        integration test for the whole model.
        """
        model = ConditionalUnet1D(**unet_config).to(device)

        output = model(**sample_data)

        assert output.shape == sample_data["sample"].shape
        assert output.dtype == sample_data["sample"].dtype

    def test_forward_pass_no_global_cond(self, unet_config, sample_data, device):
        """
        Tests that the model works correctly when the optional `global_cond`
        is not provided.
        """
        # Create a model with global_cond_dim=0 to signify no global conditioning
        config = unet_config.copy()
        config["global_cond_dim"] = 0
        model = ConditionalUnet1D(**config).to(device)

        # Remove global_cond from the input data
        data = sample_data.copy()
        data.pop("global_cond")

        # The forward pass should run without errors
        try:
            output = model(**data)
            assert output.shape == data["sample"].shape
        except Exception as e:
            pytest.fail(
                f"Forward pass without global_cond failed with an unexpected error: {e}"
            )

    def test_scalar_timestep_handling(self, unet_config, sample_data, device):
        """
        Tests that the model correctly handles a scalar integer or float for the timestep.
        """
        model = ConditionalUnet1D(**unet_config).to(device)
        data = sample_data.copy()

        # Test with a scalar integer
        data["timestep"] = 99
        output_int = model(**data)
        assert output_int.shape == data["sample"].shape

        # Test with a scalar float
        data["timestep"] = 99.0
        output_float = model(**data)
        assert output_float.shape == data["sample"].shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_portability(self, unet_config):
        """
        Tests that the entire U-Net model can be moved to a CUDA device.
        """
        # This test primarily checks if the model's .to(device) call works, which
        # implies all submodules and their parameters/buffers are moved correctly.
        model = ConditionalUnet1D(**unet_config)

        try:
            cuda_model = model.to("cuda")
            # Check a parameter from a deeply nested submodule
            assert (
                cuda_model.down_modules[0][0].conv_block1.block[0].weight.device.type
                == "cuda"
            )
        except Exception as e:
            pytest.fail(f"Moving model to CUDA failed: {e}")
