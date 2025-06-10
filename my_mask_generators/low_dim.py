from typing import Optional, Tuple

import torch

from my_mask_generators.base import BaseMaskGenerator


class LowdimMaskGenerator(BaseMaskGenerator):
    """
    Generates a mask for low-dimensional trajectories (obs, action).

    This generator creates a mask where a certain number of initial observation
    steps are marked as conditioned (visible). It can optionally also mark
    past actions as visible. This is a common setup for imitation learning
    where the policy is conditioned on a short history of observations.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ):
        """
        Args:
            action_dim: The dimensionality of the action space.
            obs_dim: The dimensionality of the observation space.
            max_n_obs_steps: The maximum number of observation steps to condition on.
            fix_obs_steps: If True, always use `max_n_obs_steps`. If False,
                           randomly sample the number of obs steps from [1, max_n_obs_steps].
            action_visible: If True, the action preceding the current observation
                            is also marked as visible.
        """
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

        # === 1. Create Dimension Masks ===
        # These masks identify which columns belong to actions and which to observations.
        is_action_dim = torch.zeros(D, dtype=torch.bool, device=self.device)
        is_action_dim[: self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # === 2. Determine Observation Timesteps ===
        # For each sample in the batch, determine how many obs steps are visible.
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

        # === 3. Create Time-Based Masks ===
        # Create a boolean mask indicating which timesteps are visible for obs and actions.
        time_idxs = torch.arange(T, device=self.device).unsqueeze(0)  # Shape: (1, T)

        # Obs are visible up to `obs_steps`
        obs_time_mask = time_idxs < obs_steps.unsqueeze(-1)  # Shape: (B, T)

        # Actions are visible up to `obs_steps - 1`
        action_time_mask = torch.zeros_like(obs_time_mask)
        if self.action_visible:
            action_steps = (obs_steps - 1).clamp(min=0)
            action_time_mask = time_idxs < action_steps.unsqueeze(-1)

        # === 4. Combine Dimension and Time Masks ===
        # Expand time masks to the full trajectory shape and combine with dim masks.
        obs_mask = obs_time_mask.unsqueeze(-1) & is_obs_dim
        action_mask = action_time_mask.unsqueeze(-1) & is_action_dim

        # The final mask is the union of the observation and action masks.
        final_mask = obs_mask | action_mask
        return final_mask
