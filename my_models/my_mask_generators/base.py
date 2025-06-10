import abc
from typing import Optional, Tuple

import torch
from torch import nn


class BaseMaskGenerator(nn.Module, abc.ABC):
    """
    Abstract base class for conditioning mask generators.

    All mask generators should inherit from this class and implement the
    `forward` method. Inheriting from `nn.Module` allows PyTorch to manage
    the device placement of any internal tensors.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    @torch.no_grad()
    def forward(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generates a boolean conditioning mask.

        Args:
            shape: The desired shape of the mask, typically (B, T, D), where
                   B=batch size, T=horizon, D=feature dimension.
            generator: A PyTorch random number generator for reproducibility.

        Returns:
            A boolean tensor of the given shape. `True` indicates a
            conditioned (known) value, while `False` indicates an unconditioned
            (to be predicted) value.
        """
        raise NotImplementedError

    def __call__(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Provides a standard callable interface."""
        return self.forward(shape, generator)


class DummyMaskGenerator(BaseMaskGenerator):
    """
    A simple generator that creates a mask where all elements are considered
    conditioned (visible). This is useful for unconditional generation or
    debugging.
    """

    def forward(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return torch.ones(size=shape, dtype=torch.bool, device=self.device)


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


if __name__ == "__main__":
    # --- DEMONSTRATION OF USAGE ---
    print("--- LowdimMaskGenerator Demo ---")
    lowdim_gen = LowdimMaskGenerator(
        action_dim=7, obs_dim=10, max_n_obs_steps=2, action_visible=True
    )
    mask_shape = (4, 16, 17)  # B, T, D
    lowdim_mask = lowdim_gen(mask_shape)

    print(f"Shape: {lowdim_mask.shape}")
    # Verify a known visible point: batch 0, time 0, obs dimension
    print(f"B0, T0, obs_dim is visible: {lowdim_mask[0, 0, 8].item()}")
    # Verify a known visible point: batch 0, time 0, action dimension
    print(f"B0, T0, act_dim is visible: {lowdim_mask[0, 0, 1].item()}")
    # Verify a known invisible point: batch 0, time 3, obs dimension
    print(f"B0, T3, obs_dim is visible: {lowdim_mask[0, 3, 8].item()}")

    print("\n--- KeypointMaskGenerator Demo ---")
    keypoint_gen = KeypointMaskGenerator(
        action_dim=7,
        keypoint_dim=2,
        n_keypoints=5,  # obs_dim = 10
        max_n_obs_steps=3,
        keypoint_visible_rate=0.5,
        context_dim=4,
        n_context_steps=1,
        action_visible=True,
    )
    mask_shape = (4, 16, 21)  # D = 7 (act) + 10 (kp) + 4 (ctx)
    keypoint_mask = keypoint_gen(mask_shape)
    print(f"Shape: {keypoint_mask.shape}")
    # Verify a known visible point: batch 0, time 0, context dim
    print(f"B0, T0, ctx_dim is visible: {keypoint_mask[0, 0, 18].item()}")
    # Verify a known invisible point: batch 0, time 2, context dim
    print(f"B0, T2, ctx_dim is visible: {keypoint_mask[0, 2, 18].item()}")
