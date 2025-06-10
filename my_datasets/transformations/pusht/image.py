from typing import Dict

import numpy as np

from models.linear_normalizer import LinearNormalizer
from my_datasets.transformations.base import BaseTransform
from my_datasets.utils.replay_buffer import ReplayBuffer


class PushTImageTransform(BaseTransform):
    """Transforms raw PushT data into image-based (multi-modal) observations."""

    def __call__(
        self, sample: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        agent_pos = sample["state"][:, :2].astype(np.float32)
        image = np.moveaxis(sample["img"], -1, 1).astype(np.float32) / 255.0

        return {
            "obs": {"image": image, "agent_pos": agent_pos},
            "action": sample["action"].astype(np.float32),
        }

    def get_normalizer(self, replay_buffer: ReplayBuffer) -> "LinearNormalizer":
        """
        Creates and fits a normalizer for the data modalities that require
        statistical normalization (e.g., actions, agent positions).
        """
        # 1. Prepare a dictionary of the data arrays you want to normalize.
        #    Notice 'image' is excluded, as its range normalization (to [0,1])
        #    is already handled directly within this transform's __call__ method.
        data_to_normalize = {
            "action": replay_buffer["action"],
            "agent_pos": replay_buffer["state"][:, :2],
        }

        # 2. Instantiate the LinearNormalizer.
        normalizer = LinearNormalizer()

        # 3. Fit the normalizer to the data. This is the crucial step.
        #    The .fit() method will compute and store the 'scale' and 'offset'
        #    for both 'action' and 'agent_pos' internally.
        normalizer.fit(data_to_normalize, mode="gaussian")

        # 4. Return the fitted normalizer. It is now ready to be used.
        return normalizer
