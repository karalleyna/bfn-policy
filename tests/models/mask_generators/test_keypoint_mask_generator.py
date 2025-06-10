from typing import Any, Dict

import pytest
import torch

# Assuming the generator classes are in a file named `my_mask_generators.py`
from models.mask_generators.keypoint import KeypointMaskGenerator

# =========================== Test Fixture (Reusable Setups) ===========================


@pytest.fixture
def shape() -> tuple:
    """Provides a standard shape (Batch, Time, Dimension) for tests."""
    return (4, 16, 17)  # B, T, D


@pytest.fixture
def generator() -> torch.Generator:
    """Provides a seeded torch.Generator for reproducible randomness."""
    return torch.Generator().manual_seed(42)


# =========================== Unit Test Classes ===========================


class TestKeypointMaskGenerator:
    """Tests for the KeypointMaskGenerator."""

    @pytest.fixture
    def keypoint_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for the keypoint generator."""
        return {
            "action_dim": 7,
            "keypoint_dim": 2,
            "n_keypoints": 5,
            "max_n_obs_steps": 3,
            "context_dim": 4,
            "n_context_steps": 1,
        }

    @pytest.fixture
    def keypoint_shape(self, keypoint_config) -> tuple:
        """Calculates the trajectory shape based on the config."""
        D = (
            keypoint_config["action_dim"]
            + (keypoint_config["n_keypoints"] * keypoint_config["keypoint_dim"])
            + keypoint_config["context_dim"]
        )
        return (4, 16, D)  # B, T, D

    def test_context_and_obs_visibility(self, keypoint_config, keypoint_shape):
        """
        Tests that the base context and observation steps are correctly masked.
        """
        gen = KeypointMaskGenerator(**keypoint_config)
        mask = gen(keypoint_shape)

        B, T, D = keypoint_shape
        action_dim = keypoint_config["action_dim"]
        context_dim = keypoint_config["context_dim"]
        n_context_steps = keypoint_config["n_context_steps"]
        n_obs_steps = keypoint_config["max_n_obs_steps"]

        # Context should only be visible for the first `n_context_steps`
        assert torch.all(mask[:, :n_context_steps, -context_dim:]).item()
        assert not torch.any(mask[:, n_context_steps:, -context_dim:]).item()

        # Keypoints (obs) should be visible for the first `n_obs_steps`
        keypoint_start_idx = action_dim
        keypoint_end_idx = -context_dim if context_dim > 0 else D
        obs_mask_part = mask[:, :, keypoint_start_idx:keypoint_end_idx]
        assert torch.all(obs_mask_part[:, :n_obs_steps, :]).item()
        assert not torch.any(obs_mask_part[:, n_obs_steps:, :]).item()

    def test_keypoint_time_independent_masking(
        self, keypoint_config, keypoint_shape, generator
    ):
        """
        Tests random keypoint masking where visibility can change over time.
        """
        gen = KeypointMaskGenerator(
            **keypoint_config,
            keypoint_visible_rate=0.5,
            time_independent_keypoints=True,
        )
        mask = gen(keypoint_shape, generator=generator)

        action_dim = keypoint_config["action_dim"]
        context_dim = keypoint_config["context_dim"]

        keypoint_start_idx = action_dim
        keypoint_end_idx = -context_dim if context_dim > 0 else keypoint_shape[2]
        obs_mask_part = mask[0, :2, keypoint_start_idx:keypoint_end_idx]

        assert torch.any(obs_mask_part).item()
        assert not torch.all(obs_mask_part).item()
        assert not torch.all(obs_mask_part[0] == obs_mask_part[1]).item()

    def test_keypoint_time_constant_masking(
        self, keypoint_config, keypoint_shape, generator
    ):
        """
        Tests random keypoint masking where visibility is constant across time.
        """
        gen = KeypointMaskGenerator(
            **keypoint_config,
            keypoint_visible_rate=0.5,
            time_independent_keypoints=False,
        )
        mask = gen(keypoint_shape, generator=generator)

        action_dim = keypoint_config["action_dim"]
        context_dim = keypoint_config["context_dim"]

        keypoint_start_idx = action_dim
        keypoint_end_idx = -context_dim if context_dim > 0 else keypoint_shape[2]
        obs_mask_part = mask[0, :2, keypoint_start_idx:keypoint_end_idx]

        torch.testing.assert_close(obs_mask_part[0], obs_mask_part[1])
