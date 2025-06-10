from typing import Optional, Tuple

import torch
from my_mask_generators.base import BaseMaskGenerator


class KeypointMaskGenerator(BaseMaskGenerator):
    """
    Generates a complex mask for trajectories involving actions, keypoints,
    and other context features.

    This is useful for policies that might be conditioned on a sparse, randomly
    sampled subset of keypoints from the observation history, in addition to
    past actions and a fixed context.
    """

    def __init__(
        self,
        action_dim: int,
        keypoint_dim: int,
        max_n_obs_steps: int,
        fix_obs_steps: bool = True,
        keypoint_visible_rate: float = 1.0,
        time_independent_keypoints: bool = False,
        action_visible: bool = False,
        context_dim: int = 0,
        n_context_steps: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.keypoint_dim = keypoint_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.keypoint_visible_rate = keypoint_visible_rate
        self.time_independent_keypoints = time_independent_keypoints
        self.action_visible = action_visible
        self.context_dim = context_dim
        self.n_context_steps = n_context_steps

    def forward(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        B, T, D = shape

        keypoint_feature_dim = D - self.action_dim - self.context_dim
        if keypoint_feature_dim % self.keypoint_dim != 0:
            raise ValueError(
                "Total keypoint feature dim is not divisible by keypoint_dim."
            )
        n_keypoints = keypoint_feature_dim // self.keypoint_dim

        # === 1. Dimension Masks ===
        is_action = torch.zeros(D, dtype=torch.bool, device=self.device)
        is_action[: self.action_dim] = True

        is_context = torch.zeros(D, dtype=torch.bool, device=self.device)
        if self.context_dim > 0:
            is_context[-self.context_dim :] = True

        is_keypoint = ~(is_action | is_context)

        # === 2. Observation Timesteps ===
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

        # === 3. Create Component Masks ===
        final_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        time_idxs = torch.arange(T, device=self.device).unsqueeze(0)  # Shape: (1, T)

        # Context Mask
        if self.context_dim > 0:
            context_time_mask = time_idxs < self.n_context_steps  # Shape: (1, T)
            final_mask |= context_time_mask.unsqueeze(-1) & is_context

        # Action Mask
        if self.action_visible:
            action_steps = (obs_steps - 1).clamp(min=0)
            action_time_mask = time_idxs < action_steps.unsqueeze(-1)  # Shape: (B, T)
            final_mask |= action_time_mask.unsqueeze(-1) & is_action

        # Keypoint Mask (most complex)
        keypoint_time_mask = time_idxs < obs_steps.unsqueeze(-1)  # Shape: (B, T)

        # Generate random visibility mask for keypoints
        if self.keypoint_visible_rate < 1.0:
            if self.time_independent_keypoints:
                # Each keypoint at each time step is independently visible
                kp_vis_shape = (B, T, n_keypoints)
            else:
                # Keypoint visibility is constant across time for each batch element
                kp_vis_shape = (B, n_keypoints)

            visible_kps = (
                torch.rand(size=kp_vis_shape, generator=generator, device=self.device)
                < self.keypoint_visible_rate
            )

            # Expand from (B, T, n_kps) to (B, T, n_kps * kp_dim)
            visible_kp_dims = torch.repeat_interleave(
                visible_kps, repeats=self.keypoint_dim, dim=-1
            )

            # Align with full trajectory shape
            if not self.time_independent_keypoints:
                visible_kp_dims = visible_kp_dims.unsqueeze(
                    1
                )  # Add time dim for broadcasting

            keypoint_time_mask = keypoint_time_mask & visible_kp_dims

        final_mask |= keypoint_time_mask.unsqueeze(-1) & is_keypoint

        return final_mask
