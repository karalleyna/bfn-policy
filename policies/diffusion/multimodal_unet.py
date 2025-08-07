"""
A concrete implementation of a diffusion policy that uses a dedicated vision
encoder for image observations and a UNet for the denoising network.
"""

from typing import Any, Dict

import torch
from torch import nn

from policies.diffusion.base import AbstractDiffusionPolicy

# Assuming this utility exists for applying a function to each tensor in a dict.
from utils import dict_apply


class VisionUnetDiffusionPolicy(AbstractDiffusionPolicy):
    """
    A diffusion policy for environments with multimodal observations.

    This policy uses a dedicated `obs_encoder` (e.g., a ResNet) to process
    observations. It supports two main conditioning modes:
    1. `obs_as_global_cond=True`: Observation features are a single global
       conditioning vector. The diffusion process runs only on the action
       sequence. This is more efficient.
    2. `obs_as_global_cond=False`: Observation features are concatenated with
       actions to form a multimodal trajectory `[action, obs_feature]`. The
       diffusion process runs on this combined sequence.
    """

    def __init__(
        self,
        denoising_network: nn.Module,
        obs_encoder: nn.Module,
        shape_meta: dict,
        n_obs_steps: int,
        obs_as_global_cond: bool = True,
        **kwargs,
    ):
        super().__init__(denoising_network=denoising_network, **kwargs)
        self.obs_encoder = obs_encoder
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.obs_feature_dim = self.obs_encoder.output_dim

    @property
    def device(self) -> torch.device:
        """
        Gets the device of the policy's components.

        This property is defined here to ensure it's available to the class,
        resolving potential inheritance issues in different environments. It
        relies on the parameters of the underlying nn.Module components.
        """
        return next(self.parameters()).device

    def _get_obs_features(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Helper function to encode observations."""
        obs_dict_reshaped = dict_apply(obs_dict, lambda x: x.reshape(-1, *x.shape[2:]))
        return self.obs_encoder(obs_dict_reshaped)

    def _prepare_inference_data(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Prepares observation data for action prediction."""
        normalized_obs = self.normalizer["obs"].normalize(obs_dict)
        B = next(iter(normalized_obs.values())).shape[0]
        obs_features_flat = self._get_obs_features(normalized_obs)

        data = dict()
        if self.obs_as_global_cond:
            data["global_cond"] = obs_features_flat.reshape(B, -1)
            data["condition_data"] = torch.zeros(
                (B, self.horizon, self.action_dim), device=self.device
            )
            data["condition_mask"] = torch.zeros_like(
                data["condition_data"], dtype=torch.bool
            )
        else:
            data["global_cond"] = None
            obs_features = obs_features_flat.reshape(
                B, self.n_obs_steps, self.obs_feature_dim
            )
            cond_data = torch.zeros(
                (B, self.horizon, self.action_dim + self.obs_feature_dim),
                device=self.device,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, : self.n_obs_steps, self.action_dim :] = obs_features
            cond_mask[:, : self.n_obs_steps, self.action_dim :] = True
            data["condition_data"] = cond_data
            data["condition_mask"] = cond_mask

        return data

    def _prepare_training_data(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Prepares a training batch for loss computation."""
        nobs = self.normalizer["obs"].normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        B, T_train, _ = nactions.shape

        data = dict()
        if self.obs_as_global_cond:
            context_obs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps])
            obs_features_flat = self._get_obs_features(context_obs)
            data["global_cond"] = obs_features_flat.reshape(B, -1)
            data["trajectory"] = nactions
        else:
            data["global_cond"] = None
            obs_features_flat = self._get_obs_features(nobs)
            obs_features = obs_features_flat.reshape(B, T_train, self.obs_feature_dim)
            data["trajectory"] = torch.cat([nactions, obs_features], dim=-1)

        return data
