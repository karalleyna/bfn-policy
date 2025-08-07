"""Defines the abstract base class for all diffusion-based policies."""

import abc
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from policies.base import AbstractPolicy


class AbstractDiffusionPolicy(AbstractPolicy, metaclass=abc.ABCMeta):
    """
    Abstract base class for diffusion-based policies.

    This class provides the core scaffolding for policies that model actions
    as a conditional denoising diffusion process. It is responsible for:
    - The high-level inference loop (`predict_action`).
    - The core denoising sampling loop (`_conditional_sample`).
    - The training loss calculation (`compute_loss`).

    Subclasses MUST implement two methods to map their specific data formats
    to the generic `trajectory` format used by the diffusion process:
    1. `_prepare_inference_data`: To map real-time observations into the
       `condition_data` and `condition_mask` tensors.
    2. `_prepare_training_data`: To map a training batch into the full
       `trajectory` that the model is trained to denoise.
    """

    def __init__(
        self,
        denoising_network: torch.nn.Module,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        action_dim: int,
        n_action_steps: int,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.denoising_network = denoising_network
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    @abc.abstractmethod
    def _prepare_inference_data(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Prepare conditioning data for inference."""
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_training_data(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Prepare a training batch for loss computation."""
        raise NotImplementedError

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generates an action by running the conditional diffusion sampler."""
        inference_data = self._prepare_inference_data(obs_dict)
        sampled_trajectory = self._conditional_sample(
            condition_data=inference_data["condition_data"],
            condition_mask=inference_data["condition_mask"],
            global_cond=inference_data.get("global_cond", None),
        )
        action_part = sampled_trajectory[..., : self.action_dim]
        unnormalized_action_pred = self.normalizer["action"].unnormalize(action_part)
        action = unnormalized_action_pred[:, : self.n_action_steps]
        return {"action": action, "action_pred": unnormalized_action_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the diffusion loss for a training batch."""
        training_data = self._prepare_training_data(batch)
        trajectory = training_data["trajectory"]
        global_cond = training_data.get("global_cond", None)

        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = self.denoising_network(
            sample=noisy_trajectory, timestep=timesteps, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type: {pred_type}")

        return F.mse_loss(pred, target)

    def _conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Performs the core denoising loop to generate a trajectory."""
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
            model_output = self.denoising_network(
                sample=noisy_trajectory, timestep=ts, global_cond=global_cond
            )
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **self.kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory
