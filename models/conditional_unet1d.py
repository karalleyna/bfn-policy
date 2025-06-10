from typing import List, Optional, Union

import einops
import torch
import torch.nn as nn

from models.embeddings.sinusoidal import SinusoidalTimestepEmbedding
from models.unet1d_components import Conv1dBlock, Downsample1d, Upsample1d


class ConditionalResidualBlock1D(nn.Module):
    """
    A residual block with conditional modulation (FiLM).

    This block applies two 1D convolutional blocks and uses a conditioning
    vector to modulate the intermediate activation, allowing the network's
    behavior to be influenced by external context like timesteps or observations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            cond_dim: Dimensionality of the conditioning vector.
            kernel_size: Size of the convolutional kernel.
            n_groups: Number of groups for Group Normalization.
            cond_predict_scale: If True, the conditioning vector predicts both a
                                scale and a bias (FiLM). If False, it only
                                predicts a bias.
        """
        super().__init__()

        self.conv_block1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.conv_block2 = Conv1dBlock(
            out_channels, out_channels, kernel_size, n_groups
        )

        # The conditioning vector is projected to produce modulation parameters.
        # This is the FiLM (Feature-wise Linear Modulation) layer.
        self.cond_predict_scale = cond_predict_scale
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.film_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, cond_channels)
        )

        # Residual connection projection
        self.residual_projection = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C_in, T).
            cond: Conditioning vector of shape (B, cond_dim).

        Returns:
            Output tensor of shape (B, C_out, T).
        """
        # First convolutional block
        out = self.conv_block1(x)

        # Generate and apply FiLM modulation
        film_params = self.film_projection(cond).unsqueeze(-1)  # -> (B, C_film, 1)
        if self.cond_predict_scale:
            scale, bias = film_params.chunk(2, dim=1)
            out = out * (scale + 1.0) + bias  # Add 1.0 to scale for stability
        else:
            bias = film_params
            out = out + bias

        # Second convolutional block and residual connection
        out = self.conv_block2(out)
        out = out + self.residual_projection(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    A 1D U-Net architecture that is conditioned on global context.

    This model is designed to denoise 1D sequences (e.g., action trajectories)
    and is the core of many diffusion policy implementations. Its behavior is
    conditioned on a global vector derived from diffusion timesteps and/or
    environmental observations.
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        down_dims: List[int],
        diffusion_step_embed_dim: int = 128,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()

        # --- 1. Condition Encoder ---
        # It processes the diffusion timestep and any global observation features.
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalTimestepEmbedding(embedding_dim=diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        # The total conditioning dimension for the FiLM layers
        total_cond_dim = diffusion_step_embed_dim + global_cond_dim

        # --- 2. U-Net Architecture ---
        all_dims = [input_dim] + down_dims
        in_out_dims = list(zip(all_dims[:-1], all_dims[1:]))

        # Shared arguments for the residual blocks
        block_kwargs = {
            "cond_dim": total_cond_dim,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "cond_predict_scale": cond_predict_scale,
        }

        # --- Encoder (Downsampling path) ---
        self.down_modules = nn.ModuleList()
        for in_dim, out_dim in in_out_dims:
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(in_dim, out_dim, **block_kwargs),
                        ConditionalResidualBlock1D(out_dim, out_dim, **block_kwargs),
                        Downsample1d(out_dim),
                    ]
                )
            )

        # --- Middle (Bottleneck) ---
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, **block_kwargs),
                ConditionalResidualBlock1D(mid_dim, mid_dim, **block_kwargs),
            ]
        )

        # --- Decoder (Upsampling path) ---
        self.up_modules = nn.ModuleList()
        for in_dim, out_dim in reversed(in_out_dims):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(out_dim * 2, in_dim, **block_kwargs),
                        ConditionalResidualBlock1D(in_dim, in_dim, **block_kwargs),
                        Upsample1d(in_dim),
                    ]
                )
            )

        # --- Final Projection Layer ---
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoises a sample sequence.

        Args:
            sample: The noisy input sequence. Shape (B, T, D_in).
            timestep: The current diffusion timestep(s). Shape (B,) or scalar.
            global_cond: Global conditioning vector. Shape (B, D_cond).

        Returns:
            The predicted denoised sequence. Shape (B, T, D_in).
        """
        # 1. Prepare inputs
        # Transpose to (B, D, T) for 1D convolutions
        sample = einops.rearrange(sample, "b t d -> b d t")

        # Ensure timestep is a 1D tensor
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
        timesteps = timestep.expand(sample.shape[0])

        # 2. Encode conditioning signals
        time_embedding = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            # Concatenate time embedding with global observation features
            cond = torch.cat([time_embedding, global_cond], dim=-1)
        else:
            cond = time_embedding

        # 3. U-Net Forward Pass
        residuals = []
        x = sample

        # Downsampling path
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            residuals.append(x)
            x = downsample(x)

        # Middle (bottleneck)
        for mid_block in self.mid_modules:
            x = mid_block(x, cond)

        # Upsampling path
        for resnet1, resnet2, upsample in self.up_modules:
            # Concatenate with residual from corresponding encoder layer
            residual = residuals.pop()
            x = torch.cat([x, residual], dim=1)

            x = resnet1(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        # 4. Final projection
        output = self.final_conv(x)

        # Transpose back to (B, T, D)
        output = einops.rearrange(output, "b d t -> b t d")
        return output


# ============================== EXAMPLE USAGE ==============================
if __name__ == "__main__":
    # --- Configuration ---
    batch_size = 4
    horizon = 16
    input_dim = 10  # e.g., action dimension
    global_cond_dim = 64  # e.g., output from an observation encoder
    down_dims = [256, 512]  # Defines 2 downsampling/upsampling stages

    # --- Model Initialization ---
    model = ConditionalUnet1D(
        input_dim=input_dim, global_cond_dim=global_cond_dim, down_dims=down_dims
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Dummy Data ---
    noisy_actions = torch.randn(batch_size, horizon, input_dim)
    timesteps = torch.randint(0, 100, (batch_size,))
    obs_features = torch.randn(batch_size, global_cond_dim)

    # --- Forward Pass ---
    predicted_noise = model(
        sample=noisy_actions, timestep=timesteps, global_cond=obs_features
    )

    # --- Verify Output ---
    print(f"\nInput shape:  {noisy_actions.shape}")
    print(f"Output shape: {predicted_noise.shape}")
    assert predicted_noise.shape == noisy_actions.shape
    print("\nConditionalUnet1D test passed.")
