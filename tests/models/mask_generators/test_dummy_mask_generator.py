import pytest
import torch

# Assuming the generator classes are in a file named `my_mask_generators.py`
from my_models.my_mask_generators.dummy import DummyMaskGenerator

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


class TestDummyMaskGenerator:
    """Tests for the DummyMaskGenerator."""

    def test_returns_all_true_mask(self, shape):
        """Tests that the generator returns a mask of all True values."""
        gen = DummyMaskGenerator()
        mask = gen(shape)

        assert mask.shape == shape
        assert mask.dtype == torch.bool
        assert torch.all(mask).item()
