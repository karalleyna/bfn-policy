from typing import Dict

import numpy as np
import pytest

from datasets.transformations.pusht.low_dim import PushTLowDimTransform


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


class TestPushTLowDimTransform:
    """A suite of unit tests for the PushTLowDimTransform class."""

    def test_output_structure_and_keys(self, raw_pusht_sample):
        """
        Tests that the transform produces a flat dictionary with 'obs' and 'action' keys.
        """
        transform = PushTLowDimTransform()
        transformed_data = transform(raw_pusht_sample)

        assert isinstance(transformed_data, dict)
        assert set(transformed_data.keys()) == {"obs", "action"}
        assert not isinstance(
            transformed_data["obs"], dict
        )  # Obs should be a single tensor

    def test_output_shapes(self, raw_pusht_sample):
        """
        Tests that the observation vector has the correct concatenated shape.
        """
        transform = PushTLowDimTransform()
        transformed_data = transform(raw_pusht_sample)

        T = raw_pusht_sample["state"].shape[0]

        # Expected obs dim = (num_keypoints * 2) + 2 (for agent_pos)
        expected_obs_dim = raw_pusht_sample["keypoint"].shape[1] * 2 + 2
        expected_obs_shape = (T, expected_obs_dim)

        assert transformed_data["obs"].shape == expected_obs_shape
        assert transformed_data["action"].shape == raw_pusht_sample["action"].shape

    def test_concatenation_logic(self):
        """
        Tests the concatenation logic with known values to ensure correctness.
        """
        transform = PushTLowDimTransform()

        # Create a sample with simple, known values
        raw_sample = {
            "keypoint": np.array([[[1, 2], [3, 4]]]),  # T=1, K=2
            "state": np.array([[10, 11, 12]]),  # T=1, agent_pos is [10, 11]
            "action": np.array([[100, 101]]),
        }

        transformed_data = transform(raw_sample)

        # Manually compute the expected observation vector
        # Flattened keypoints are [1, 2, 3, 4]. Agent pos is [10, 11].
        expected_obs = np.array([[1, 2, 3, 4, 10, 11]], dtype=np.float32)

        np.testing.assert_array_equal(transformed_data["obs"], expected_obs)

    def test_output_dtypes(self, raw_pusht_sample):
        """
        Tests that all output arrays are of type float32.
        """
        transform = PushTLowDimTransform()
        transformed_data = transform(raw_pusht_sample)

        assert transformed_data["obs"].dtype == np.float32
        assert transformed_data["action"].dtype == np.float32
