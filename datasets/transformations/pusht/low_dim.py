from typing import Dict

import numpy as np

from datasets.transformations.base import BaseTransform
from datasets.utils.replay_buffer import ReplayBuffer
from models.normalizers.linear import LinearNormalizer


class PushTLowDimTransform(BaseTransform):
    """Transforms raw PushT data into low-dimensional observations."""

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        agent_pos = sample["state"][:, :2]
        keypoints = sample["keypoint"]
        obs = np.concatenate(
            [keypoints.reshape(keypoints.shape[0], -1), agent_pos],
            axis=-1,
            dtype=np.float32,
        )
        return {"obs": obs, "action": sample["action"].astype(np.float32)}

    def get_normalizer(self, replay_buffer: ReplayBuffer) -> "LinearNormalizer":
        """
        Creates and fits a normalizer for the transformed low-dimensional data.
        """
        # 1. First, create a dictionary of the raw data arrays needed by the transform.
        raw_data = {
            "state": replay_buffer["state"],
            "keypoint": replay_buffer["keypoint"],
            "action": replay_buffer["action"],
        }

        # 2. Apply the transform to the raw data to get the processed data
        #    that the policy will actually see (e.g., {'obs': ..., 'action': ...}).
        data_to_normalize = self(raw_data)

        # 3. Instantiate the LinearNormalizer.
        normalizer = LinearNormalizer()

        # 4. CRITICAL STEP: Fit the normalizer to the processed data.
        #    This computes the mean/std or min/max and stores them inside the object.
        normalizer.fit(data_to_normalize, mode="gaussian")

        # 5. Return the fully fitted normalizer.
        return normalizer
