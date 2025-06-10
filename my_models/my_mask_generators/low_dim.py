from typing import Optional, Tuple

import torch

from my_models.my_mask_generators.base import BaseMaskGenerator


class LowdimMaskGenerator(BaseMaskGenerator):
    """Generates a mask for low-dimensional trajectories (obs, action)."""

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    def forward(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        B, T, D = shape
        if D != (self.action_dim + self.obs_dim):
            raise ValueError("Feature dimension D does not match action_dim + obs_dim.")

        is_action_dim = torch.zeros(D, dtype=torch.bool, device=self.device)
        is_action_dim[: self.action_dim] = True
        is_obs_dim = ~is_action_dim

        if self.fix_obs_steps:
            obs_steps = torch.full(
                (B,), fill_value=self.max_n_obs_steps, device=self.device
            )
        else:
            obs_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,
                size=(B,),
                generator=generator,
                device=self.device,
            )

        time_idxs = torch.arange(T, device=self.device).unsqueeze(0)
        obs_time_mask = time_idxs < obs_steps.unsqueeze(-1)

        action_time_mask = torch.zeros_like(obs_time_mask)
        if self.action_visible:
            action_steps = (obs_steps - 1).clamp(min=0)
            action_time_mask = time_idxs < action_steps.unsqueeze(-1)

        obs_mask = obs_time_mask.unsqueeze(-1) & is_obs_dim
        # --- FIX IS HERE: Corrected variable name from `is_action` to `is_action_dim` ---
        action_mask = action_time_mask.unsqueeze(-1) & is_action_dim

        return obs_mask | action_mask
