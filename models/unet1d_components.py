from typing import Optional

import torch
from torch import nn


class Conv1dBlock(nn.Module):
    """
    A standard 1D convolutional block consisting of a 1D convolution,
    Group Normalization, and a SiLU (Swish) activation function.

    This block is a fundamental building block for U-Net architectures,
    designed to process sequential data while maintaining channel information.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ):
        """
        Initializes the Conv1dBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: The size of the 1D convolutional kernel.
            n_groups: Number of groups for Group Normalization. This should be
                      a divisor of `out_channels`.
        """
        super().__init__()

        # Using padding='same' with an odd kernel_size ensures the output
        # sequence length is the same as the input.
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to use padding='same'")

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(),  # SiLU (Swish) is a modern replacement for Mish
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through the convolutional block.

        Args:
            x: Input tensor of shape (B, C_in, T), where B is the batch size,
               C_in is the number of input channels, and T is the sequence length.

        Returns:
            Output tensor of shape (B, C_out, T).
        """
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    A residual block that incorporates two Conv1dBlocks and a residual connection.

    Residual connections are crucial for training deep neural networks by
    allowing gradients to flow more easily through the network.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ):
        super().__init__()

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)

        # If the number of input channels is different from the output channels,
        # a 1x1 convolution is used to project the residual connection to the
        # correct dimension.
        if in_channels != out_channels:
            self.residual_projection = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the residual block.

        Args:
            x: Input tensor of shape (B, C_in, T).

        Returns:
            Output tensor of shape (B, C_out, T).
        """
        residual = self.residual_projection(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class Downsample1d(nn.Module):
    """
    A 1D downsampling module that reduces the sequence length by a factor of 2.
    """

    def __init__(self, channels: int):
        """
        Initializes the Downsample1d module.

        Args:
            channels: The number of input and output channels.
        """
        super().__init__()
        # A 1D convolution with a stride of 2 halves the sequence length.
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsamples the input tensor.

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            Output tensor of shape (B, C, T/2).
        """
        return self.conv(x)


class Upsample1d(nn.Module):
    """
    A 1D upsampling module that doubles the sequence length.
    """

    def __init__(self, channels: int):
        """
        Initializes the Upsample1d module.

        Args:
            channels: The number of input and output channels.
        """
        super().__init__()
        # A 1D transposed convolution with a stride of 2 doubles the sequence length.
        # kernel_size=4, stride=2, padding=1 is a standard configuration for this.
        self.conv = nn.ConvTranspose1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsamples the input tensor.

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            Output tensor of shape (B, C, T*2).
        """
        return self.conv(x)


# ============================== EXAMPLE USAGE ==============================
if __name__ == "__main__":
    batch_size = 4
    sequence_length = 32

    # --- Test Conv1dBlock ---
    print("--- Testing Conv1dBlock ---")
    in_channels_conv = 64
    out_channels_conv = 128
    conv_block = Conv1dBlock(in_channels_conv, out_channels_conv, kernel_size=3)
    x_conv = torch.randn(batch_size, in_channels_conv, sequence_length)
    y_conv = conv_block(x_conv)
    print(f"Input shape: {x_conv.shape}")
    print(f"Output shape: {y_conv.shape}")
    assert y_conv.shape == (batch_size, out_channels_conv, sequence_length)
    print("Conv1dBlock test passed.")

    # --- Test ResidualBlock ---
    print("\n--- Testing ResidualBlock ---")
    in_channels_res = 128
    out_channels_res = 128
    res_block = ResidualBlock(in_channels_res, out_channels_res, kernel_size=5)
    x_res = torch.randn(batch_size, in_channels_res, sequence_length)
    y_res = res_block(x_res)
    print(f"Input shape: {x_res.shape}")
    print(f"Output shape: {y_res.shape}")
    assert y_res.shape == x_res.shape
    print("ResidualBlock (same dim) test passed.")

    # Test residual block with dimension change
    out_channels_res_diff = 256
    res_block_diff = ResidualBlock(
        in_channels_res, out_channels_res_diff, kernel_size=3
    )
    y_res_diff = res_block_diff(x_res)
    print(f"\nInput shape: {x_res.shape}")
    print(f"Output shape (dim change): {y_res_diff.shape}")
    assert y_res_diff.shape == (batch_size, out_channels_res_diff, sequence_length)
    print("ResidualBlock (diff dim) test passed.")

    # --- Test Downsample1d ---
    print("\n--- Testing Downsample1d ---")
    channels_sample = 256
    downsampler = Downsample1d(channels=channels_sample)
    x_down = torch.randn(batch_size, channels_sample, sequence_length)
    y_down = downsampler(x_down)
    print(f"Input shape: {x_down.shape}")
    print(f"Output shape: {y_down.shape}")
    assert y_down.shape == (batch_size, channels_sample, sequence_length // 2)
    print("Downsample1d test passed.")

    # --- Test Upsample1d ---
    print("\n--- Testing Upsample1d ---")
    upsampler = Upsample1d(channels=channels_sample)
    # Use the downsampled output as input for the upsampler
    y_up = upsampler(y_down)
    print(f"Input shape: {y_down.shape}")
    print(f"Output shape: {y_up.shape}")
    # The upsampled output should have the original sequence length
    assert y_up.shape == x_down.shape
    print("Upsample1d test passed.")
