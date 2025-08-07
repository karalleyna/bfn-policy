import pytest
import torch
from torch import nn

from models.unet1d_components import (
    Conv1dBlock,
    Downsample1d,
    ResidualBlock,
    Upsample1d,
)


@pytest.fixture(params=[(4, 64, 32), (8, 128, 64)])  # (B, C, T)
def sample_data(request):
    """Provides sample tensor data with different shapes for testing."""
    batch_size, channels, seq_len = request.param
    return torch.randn(batch_size, channels, seq_len)


@pytest.fixture
def device() -> torch.device:
    """Provides a device to run tests on (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestConv1dBlock:
    """A suite of unit tests for the Conv1dBlock class."""

    def test_output_shape_and_dtype(self, sample_data, device):
        """Tests that the block produces an output with the correct shape and dtype."""
        in_channels = sample_data.shape[1]
        out_channels = in_channels * 2

        block = Conv1dBlock(in_channels, out_channels, kernel_size=3).to(device)
        x = sample_data.to(device)

        output = block(x)

        assert output.shape == (x.shape[0], out_channels, x.shape[2])
        assert output.dtype == torch.float32

    def test_invalid_kernel_size_raises_error(self):
        """Tests that an even kernel_size raises a ValueError."""
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            Conv1dBlock(64, 128, kernel_size=4)


class TestResidualBlock:
    """A suite of unit tests for the ResidualBlock class."""

    def test_output_shape_same_channels(self, sample_data, device):
        """Tests the output shape when input and output channels are the same."""
        channels = sample_data.shape[1]
        block = ResidualBlock(channels, channels, kernel_size=5).to(device)
        x = sample_data.to(device)

        output = block(x)

        # Output shape should be identical to input shape
        assert output.shape == x.shape

    def test_output_shape_different_channels(self, sample_data, device):
        """Tests the output shape when input and output channels are different."""
        in_channels = sample_data.shape[1]
        out_channels = in_channels * 2

        block = ResidualBlock(in_channels, out_channels, kernel_size=3).to(device)
        x = sample_data.to(device)

        output = block(x)

        # Output shape should have the new number of channels
        expected_shape = (x.shape[0], out_channels, x.shape[2])
        assert output.shape == expected_shape

    def test_residual_projection_creation(self):
        """Tests that the residual projection layer is created correctly."""
        # Case 1: Same channels, projection should be nn.Identity
        res_block_same = ResidualBlock(128, 128, kernel_size=3)
        assert isinstance(res_block_same.residual_projection, nn.Identity)

        # Case 2: Different channels, projection should be nn.Conv1d
        res_block_diff = ResidualBlock(128, 256, kernel_size=3)
        assert isinstance(res_block_diff.residual_projection, nn.Conv1d)


class TestDownsample1d:
    """A suite of unit tests for the Downsample1d class."""

    def test_output_shape(self, sample_data, device):
        """Tests that the sequence length is correctly halved."""
        channels = sample_data.shape[1]
        downsampler = Downsample1d(channels=channels).to(device)
        x = sample_data.to(device)

        output = downsampler(x)

        expected_shape = (x.shape[0], channels, x.shape[2] // 2)
        assert output.shape == expected_shape


class TestUpsample1d:
    """A suite of unit tests for the Upsample1d class."""

    def test_output_shape(self, sample_data, device):
        """Tests that the sequence length is correctly doubled."""
        channels = sample_data.shape[1]
        upsampler = Upsample1d(channels=channels).to(device)
        x = sample_data.to(device)

        output = upsampler(x)

        expected_shape = (x.shape[0], channels, x.shape[2] * 2)
        assert output.shape == expected_shape

    def test_downsample_upsample_restores_shape(self, sample_data, device):
        """
        An integration test to verify that applying downsampling then upsampling
        restores the original sequence length.
        """
        channels = sample_data.shape[1]
        x = sample_data.to(device)

        downsampler = Downsample1d(channels=channels).to(device)
        upsampler = Upsample1d(channels=channels).to(device)

        down_x = downsampler(x)
        up_x = upsampler(down_x)

        # The final shape should match the original input shape
        assert up_x.shape == x.shape
