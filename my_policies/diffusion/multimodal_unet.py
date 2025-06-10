import abc
from typing import Any, Dict

import torch
import torch.nn as nn
from einops import reduce

from my_policies.diffusion.base import DiffusionPolicy
from utils import dict_apply


class MultiModalDiffusionUnetPolicy(DiffusionPolicy):
    """
    Concrete implementation of a diffusion policy.
    Its main job is to prepare data for the BaseDiffusionPolicy's methods.
    """

    def __init__(self, shape_meta: dict, obs_encoder: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder

    def _prepare_obs_features(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Reshape obs from (B, T, ...) to (B*T, ...) for batch processing
        obs_dict_reshaped = dict_apply(obs_dict, lambda x: x.reshape(-1, *x.shape[2:]))
        return self.obs_encoder(obs_dict_reshaped)

    def _prepare_inference_data(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T = self.horizon
        nobs_features_flat = self._prepare_obs_features(nobs)

        data = dict()
        if self.obs_as_global_cond:
            data["global_cond"] = nobs_features_flat.reshape(
                B, self.n_obs_steps * self.obs_feature_dim
            )
            data["condition_data"] = torch.zeros(
                (B, T, self.action_dim), device=self.device
            )
            data["condition_mask"] = torch.zeros_like(
                data["condition_data"], dtype=torch.bool
            )
        else:  # Inpainting logic
            data["global_cond"] = None
            nobs_features = nobs_features_flat.reshape(
                B, self.n_obs_steps, self.obs_feature_dim
            )
            cond_data = torch.zeros(
                (B, T, self.action_dim + self.obs_feature_dim), device=self.device
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, : self.n_obs_steps, self.action_dim :] = nobs_features
            cond_mask[:, : self.n_obs_steps, self.action_dim :] = True
            data["condition_data"] = cond_data
            data["condition_mask"] = cond_mask
        return data

    def _prepare_training_data(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        B, T_train, _ = nactions.shape

        data = dict()
        if self.obs_as_global_cond:
            context_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps])
            nobs_features_flat = self._prepare_obs_features(context_nobs)
            data["global_cond"] = nobs_features_flat.reshape(
                B, self.n_obs_steps * self.obs_feature_dim
            )
            data["trajectory"] = nactions
        else:  # Inpainting logic
            data["global_cond"] = None
            nobs_features_flat = self._prepare_obs_features(nobs)
            nobs_features = nobs_features_flat.reshape(B, T_train, self.obs_feature_dim)
            data["trajectory"] = torch.cat([nactions, nobs_features], dim=-1)
        return data
