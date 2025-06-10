from typing import Optional, Tuple

import torch

from models.mask_generators.base import BaseMaskGenerator


class KeypointMaskGenerator(BaseMaskGenerator):
    """Generates a complex mask for trajectories involving keypoints."""

    def __init__(
        self,
        action_dim: int,
        keypoint_dim: int,
        n_keypoints: int,
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
        self.n_keypoints = n_keypoints
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
        if keypoint_feature_dim != (self.n_keypoints * self.keypoint_dim):
            raise ValueError("Dimension mismatch for keypoints.")

        is_action = torch.zeros(D, dtype=torch.bool, device=self.device)
        is_action[: self.action_dim] = True

        is_context = torch.zeros(D, dtype=torch.bool, device=self.device)
        if self.context_dim > 0:
            is_context[-self.context_dim :] = True
        is_keypoint = ~(is_action | is_context)

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

        final_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        time_idxs = torch.arange(T, device=self.device).unsqueeze(0)

        if self.context_dim > 0:
            context_time_mask = time_idxs < self.n_context_steps
            final_mask |= context_time_mask.unsqueeze(-1) & is_context

        if self.action_visible:
            action_steps = (obs_steps - 1).clamp(min=0)
            action_time_mask = time_idxs < action_steps.unsqueeze(-1)
            final_mask |= action_time_mask.unsqueeze(-1) & is_action

        # --- LOGIC FIX IS HERE ---
        # Create a visibility mask for the keypoint feature dimensions specifically
        obs_time_mask = time_idxs < obs_steps.unsqueeze(-1)

        kp_visibility_mask = obs_time_mask.unsqueeze(-1).expand(
            -1, -1, keypoint_feature_dim
        )

        if self.keypoint_visible_rate < 1.0:
            if self.time_independent_keypoints:
                kp_vis_shape = (B, T, self.n_keypoints)
            else:
                kp_vis_shape = (B, self.n_keypoints)

            visible_kps = (
                torch.rand(size=kp_vis_shape, generator=generator, device=self.device)
                < self.keypoint_visible_rate
            )

            visible_kp_dims = torch.repeat_interleave(
                visible_kps, repeats=self.keypoint_dim, dim=-1
            )

            if not self.time_independent_keypoints:
                visible_kp_dims = visible_kp_dims.unsqueeze(1)

            kp_visibility_mask = kp_visibility_mask & visible_kp_dims

        # Place the keypoint mask into the correct slice of the final mask
        kp_start_idx = self.action_dim
        kp_end_idx = D - self.context_dim
        final_mask[:, :, kp_start_idx:kp_end_idx] |= kp_visibility_mask

        return final_mask
