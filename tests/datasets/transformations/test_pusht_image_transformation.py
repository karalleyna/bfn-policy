from typing import Dict

import numpy as np
import pytest

from my_datasets.transformations.pusht.image import PushTImageTransform

# =========================== Test Fixtures (Reusable Setups) ===========================


@pytest.fixture
def raw_pusht_sample() -> Dict[str, np.ndarray]:
    """
    Provides a single, raw data sample dictionary with realistic shapes and types,
    mimicking the output of `replay_buffer[start:end]`.
    """
    sequence_length = 16
    return {
        "img": (np.random.rand(sequence_length, 64, 64, 3) * 255).astype(np.uint8),
        "state": np.random.randn(sequence_length, 10).astype(np.float32),
        "keypoint": np.random.randn(sequence_length, 5, 2).astype(np.float32),
        "action": np.random.randn(sequence_length, 7).astype(np.float32),
    }


# =========================== Unit Test Classes ===========================


class TestPushTImageTransform:
    """A suite of unit tests for the PushTImageTransform class."""

    def test_output_structure_and_keys(self, raw_pusht_sample):
        """
        Tests that the transform produces the correct dictionary structure
        with the expected keys for the image modality.
        """
        transform = PushTImageTransform()
        transformed_data = transform(raw_pusht_sample)

        # Check top-level keys
        assert isinstance(transformed_data, dict)
        assert set(transformed_data.keys()) == {"obs", "action"}

        # Check nested 'obs' keys
        assert isinstance(transformed_data["obs"], dict)
        assert set(transformed_data["obs"].keys()) == {"image", "agent_pos"}

    def test_output_shapes(self, raw_pusht_sample):
        """
        Tests that the output arrays have the correct shapes after transformation.
        """
        transform = PushTImageTransform()
        transformed_data = transform(raw_pusht_sample)

        T = raw_pusht_sample["img"].shape[0]  # Sequence length

        # Image shape should be (T, C, H, W)
        expected_img_shape = (T, 3, 64, 64)
        assert transformed_data["obs"]["image"].shape == expected_img_shape

        # Agent position shape should be (T, 2)
        expected_pos_shape = (T, 2)
        assert transformed_data["obs"]["agent_pos"].shape == expected_pos_shape

        # Action shape should be unchanged
        assert transformed_data["action"].shape == raw_pusht_sample["action"].shape

    def test_output_dtypes(self, raw_pusht_sample):
        """
        Tests that all output arrays are of type float32.
        """
        transform = PushTImageTransform()
        transformed_data = transform(raw_pusht_sample)

        assert transformed_data["obs"]["image"].dtype == np.float32
        assert transformed_data["obs"]["agent_pos"].dtype == np.float32
        assert transformed_data["action"].dtype == np.float32

    def test_image_normalization(self, raw_pusht_sample):
        """
        Tests that the image pixel values are correctly normalized to the [0, 1] range.
        """
        transform = PushTImageTransform()
        transformed_data = transform(raw_pusht_sample)

        normalized_image = transformed_data["obs"]["image"]

        assert np.min(normalized_image) >= 0.0
        assert np.max(normalized_image) <= 1.0
