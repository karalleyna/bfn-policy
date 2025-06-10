import pytest
import torch

# Assuming the generator classes are in a file named `my_mask_generators.py`
from my_models.my_mask_generators.low_dim import LowdimMaskGenerator


@pytest.fixture
def shape() -> tuple:
    """Provides a standard shape (Batch, Time, Dimension) for tests."""
    return (4, 16, 17)


@pytest.fixture
def generator() -> torch.Generator:
    """Provides a seeded torch.Generator for reproducible randomness."""
    return torch.Generator().manual_seed(42)


class TestLowdimMaskGenerator:
    """Tests for the LowdimMaskGenerator."""

    # ... (other tests for LowdimMaskGenerator are correct) ...

    def test_random_obs_steps(self, shape, generator):
        """
        Tests that with fix_obs_steps=False, the number of visible observation
        steps varies per batch element but stays within the valid range.
        This test is now deterministic.
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

        # --- FIX IS HERE ---
        # To get a comparable random sequence, we must create a new generator
        # with the same seed, as the one passed to `gen` has had its state advanced.
        expected_generator = torch.Generator().manual_seed(42)
        expected_obs_steps = torch.randint(
            low=1, high=max_n_obs_steps + 1, size=(B,), generator=expected_generator
        )

        obs_part = mask[:, :, action_dim:]
        # To get the number of visible steps, we can check for any True value
        # along the feature dimension, and then sum along the time dimension.
        num_visible_steps = torch.any(obs_part, dim=-1).sum(dim=-1)

        torch.testing.assert_close(num_visible_steps, expected_obs_steps)
