import pytest
import torch

# Assuming the generator classes are in a file named `my_mask_generators.py`
from my_models.my_mask_generators.low_dim import LowdimMaskGenerator

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


class TestLowdimMaskGenerator:
    """Tests for the LowdimMaskGenerator."""

    def test_fixed_obs_steps_no_action(self, shape):
        """
        Tests the primary case: a fixed number of observation steps are visible,
        actions are not visible.
        """
        B, T, D = shape
        action_dim, obs_dim = 7, 10
        n_obs_steps = 2

        gen = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        mask = gen(shape)

        assert mask.shape == shape

        # Check obs part: first n_obs_steps should be True, rest False
        assert torch.all(mask[:, :n_obs_steps, action_dim:]).item()
        assert not torch.any(mask[:, n_obs_steps:, action_dim:]).item()

        # Check action part: all should be False
        assert not torch.any(mask[:, :, :action_dim]).item()

    def test_fixed_obs_steps_with_action(self, shape):
        """Tests that when action_visible=True, past actions are also visible."""
        B, T, D = shape
        action_dim, obs_dim = 7, 10
        n_obs_steps = 3

        gen = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=True,
        )
        mask = gen(shape)

        # Obs part: first n_obs_steps (3) should be True
        assert torch.all(mask[:, :n_obs_steps, action_dim:]).item()

        # Action part: first n_obs_steps - 1 (2) should be True
        assert torch.all(mask[:, : n_obs_steps - 1, :action_dim]).item()
        assert not torch.any(mask[:, n_obs_steps - 1 :, :action_dim]).item()

    def test_random_obs_steps(self, shape, generator):
        """
        Tests that with fix_obs_steps=False, the number of visible observation
        steps varies per batch element but stays within the valid range.
        """
        B, T, D = shape
        action_dim, obs_dim = 7, 10
        max_n_obs_steps = 5

        gen = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=obs_dim,
            max_n_obs_steps=max_n_obs_steps,
            fix_obs_steps=False,
        )
        mask = gen(shape, generator=generator)

        # For each item in the batch, find the number of visible obs steps
        # by checking the last True value along the time dimension.
        obs_part = mask[:, :, action_dim:]
        num_visible_steps = (torch.cumsum(obs_part, dim=1) > 0).sum(dim=1)[:, 0]

        # Assert that all calculated step counts are between 1 and max_n_obs_steps
        assert torch.all(num_visible_steps >= 1).item()
        assert torch.all(num_visible_steps <= max_n_obs_steps).item()
        # Assert that not all are the same (probabilistically, for this seed)
        assert len(torch.unique(num_visible_steps)) > 1

    def test_dimension_mismatch_error(self):
        """Tests that a ValueError is raised if D != action_dim + obs_dim."""
        gen = LowdimMaskGenerator(action_dim=7, obs_dim=10, max_n_obs_steps=2)
        wrong_shape = (4, 16, 18)  # Should be 17

        with pytest.raises(ValueError, match="Feature dimension D does not match"):
            gen(wrong_shape)
