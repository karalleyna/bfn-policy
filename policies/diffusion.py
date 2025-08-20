from typing import Dict, Tuple

import torch
import torch.nn as nn

# Assuming diffusers is installed: pip install diffusers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from models.normalizers.base import BaseNormalizer
from policies.base import BasePolicy


class DiffusionPolicy(BasePolicy):
    """
    A general-purpose diffusion-based policy.

    This policy uses a denoising diffusion model to predict action trajectories
    based on observations. It is designed to be flexible and can be configured
    with different noise prediction networks (e.g., UNet, Transformer) and
    observation encoders.
    """

    def __init__(
        self,
        noise_prediction_net: nn.Module,
        obs_encoder: nn.Module,
        normalizer: BaseNormalizer,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        num_diffusion_iters: int = 100,
        # DDPM scheduler parameters
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Initializes the DiffusionPolicy.

        Args:
            noise_prediction_net: The network that predicts noise from a noisy
                                  action trajectory. This is the core of the
                                  diffusion model.
            obs_encoder: The network that encodes raw observations into a feature
                         vector (conditioning signal).
            normalizer: A normalizer instance for observations and actions.
            action_dim: The dimensionality of the action space.
            obs_horizon: The number of observation steps to use as context.
            pred_horizon: The number of action steps to predict.
            num_diffusion_iters: The number of denoising steps during inference.
            beta_schedule: The schedule for beta values (noise levels) in DDPM.
            beta_start: The starting value of beta.
            beta_end: The ending value of beta.
        """
        super().__init__()

        self.obs_encoder = obs_encoder
        self.noise_prediction_net = noise_prediction_net
        self.normalizer = normalizer

        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_diffusion_iters = num_diffusion_iters

        # Initialize the DDPM noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            clip_sample=True,  # Clip the sample to [-1, 1]
            prediction_type="epsilon",  # Predict the noise (epsilon)
        )

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Predicts a sequence of actions using the diffusion model.

        This method performs the full denoising process to generate an action
        trajectory from an initial noisy state.

        Args:
            obs_dict: A dictionary of observation tensors.

        Returns:
            A dictionary containing the predicted action trajectory, un-normalized.
            - 'action': Tensor of shape (B, pred_horizon, action_dim)
        """
        # 1. Normalize observations and move to the correct device
        nobs = self.normalizer.normalize(obs_dict)
        device = self.device
        nobs = {k: v.to(device) for k, v in nobs.items()}

        # 2. Encode observations to get a conditioning vector
        obs_features = self.obs_encoder(nobs)

        # 3. Initialize action trajectory with pure Gaussian noise
        B = next(iter(nobs.values())).shape[0]
        noisy_actions = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=device
        )

        # Set scheduler to the number of inference steps
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        # 4. Iteratively denoise the action trajectory
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.noise_prediction_net(
                sample=noisy_actions, timestep=t, global_cond=obs_features
            )

            # Denoise for one step
            noisy_actions = self.noise_scheduler.step(
                model_output=noise_pred, timestep=t, sample=noisy_actions
            ).prev_sample

        # 5. Un-normalize the final denoised action trajectory
        action_pred = self.normalizer.unnormalize({"action": noisy_actions})["action"]

        return {"action": action_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the diffusion loss for a batch of training data.

        Args:
            batch: A dictionary containing 'obs' and 'action' tensors.

        Returns:
            A scalar tensor representing the L2 loss between the predicted and
            actual noise.
        """
        # 1. Normalize observations and actions
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer.normalize({"action": batch["action"]})["action"]

        # 2. Encode observations
        obs_features = self.obs_encoder(nobs)

        # 3. Sample random noise
        noise = torch.randn_like(nactions)
        B = nactions.shape[0]

        # 4. Sample a random timestep for each trajectory
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
        ).long()

        # 5. Add noise to the actions to create the noisy input
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)

        # 6. Predict the noise from the noisy actions
        noise_pred = self.noise_prediction_net(
            sample=noisy_actions, timestep=timesteps, global_cond=obs_features
        )

        # 7. Compute the L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the noise prediction network.
        This is a convenience property to ensure all tensors are on the same device.
        """
        return next(self.noise_prediction_net.parameters()).device
