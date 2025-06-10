import abc
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from my_policies.base import Policy


class DiffusionPolicy(Policy):
    """
    Abstract base class for all diffusion-based policies.
    It handles the core diffusion logic (sampling, loss calculation),
    while leaving data preparation to child classes.
    """

    def __init__(
        self,
        denoising_model: nn.Module,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        obs_as_global_cond: bool = True,
        obs_feature_dim: int = 0,  # Must be provided by child if obs_as_global_cond=False
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.denoising_model = denoising_model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
        )
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        scheduler.set_timesteps(self.num_inference_steps, device=trajectory.device)
        for t in scheduler.timesteps:
            noisy_trajectory = trajectory.clone()
            noisy_trajectory[condition_mask] = condition_data[condition_mask]
            ts = torch.full(
                (trajectory.shape[0],), t, device=trajectory.device, dtype=torch.long
            )
            model_output = self.denoising_model(
                noisy_trajectory, ts, global_cond=global_cond
            )
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        data_for_sampling = self._prepare_inference_data(obs_dict)
        nsample = self.conditional_sample(
            condition_data=data_for_sampling["condition_data"],
            condition_mask=data_for_sampling["condition_mask"],
            global_cond=data_for_sampling.get("global_cond", None),
            **self.kwargs,
        )
        naction_pred = nsample[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        action = action_pred[:, : self.n_action_steps]
        return {"action": action, "action_pred": action_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        data_for_loss = self._prepare_training_data(batch)
        trajectory = data_for_loss["trajectory"]
        global_cond = data_for_loss.get("global_cond", None)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        condition_mask = self.mask_generator(trajectory.shape).to(trajectory.device)
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.denoising_model(
            noisy_trajectory, timesteps, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss_mask = ~condition_mask
        loss = loss * loss_mask.type(loss.dtype)
        return reduce(loss, "b ... -> b (...)", "mean").mean()

    @abc.abstractmethod
    def _prepare_inference_data(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_training_data(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        raise NotImplementedError
