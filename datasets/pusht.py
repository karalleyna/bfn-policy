# Standard library imports
import copy
from abc import ABC, abstractmethod
from typing import Dict, Union

# Third-party imports
import numpy as np
import torch

# Project-specific imports
from datasets.base import BaseDataset
from utils.normalizer import LinearNormalizer, get_image_range_normalizer
from utils.pytorch import dict_apply
from utils.replay_buffer import ReplayBuffer
from utils.sampler import SequenceSampler, downsample_mask, get_val_mask


# Abstract base class for PushT dataset
class BasePushTDataset(BaseDataset, ABC):
    def __init__(
        self,
        zarr_path: str,  # Path to the dataset stored in Zarr format
        keys: list,  # List of keys (e.g., ['img', 'state', 'action']) to load from the dataset
        horizon: int = 1,  # Length of each sampled sequence (number of timesteps)
        pad_before: int = 0,  # Number of padding steps to add before each sequence
        pad_after: int = 0,  # Number of padding steps to add after each sequence
        seed: int = 42,  # Seed for random operations (e.g. validation split) to ensure reproducibility
        val_ratio: float = 0.0,  # Fraction of episodes to use for validation
        max_train_episodes: int = None,  # Optional cap on the number of training episodes (None = use all)
    ):
        super().__init__()

        # Load data from Zarr file into a replay buffer
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        # Split episodes into training and validation using provided ratio and seed
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask  # Boolean negation to get training episodes

        # Optionally downsample training episodes
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        # Initialize a sampler to extract fixed-length sequences from training data
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        # Store key dataset properties
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self) -> "BasePushTDataset":
        # Create a shallow copy and replace its sampler with one using the validation mask
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Sample a sequence and convert it to model-ready tensors
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

    @abstractmethod
    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict:
        """
        Convert a raw sample into a dictionary of model inputs/outputs.

        Should return:
            {
                'obs': Tensor or Dict[str, Tensor],
                'action': Tensor,
            }
        """
        pass

    @abstractmethod
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        Return a normalizer fitted to the data (for preprocessing).
        """
        pass

    def get_all_actions(self) -> torch.Tensor:
        """
        Optional override for retrieving all actions in the dataset.
        """
        raise NotImplementedError()


# Dataset for PushT with low-dimensional (keypoint + state) observations
class PushTLowDimDataset(BasePushTDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key="keypoint",
        state_key="state",
        action_key="action",
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        # Save keys to access specific parts of the dataset
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key

        keys = [obs_key, state_key, action_key]

        # Initialize the base class
        super().__init__(
            zarr_path=zarr_path,
            keys=keys,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
        )

    def _sample_to_data(self, sample):
        # Extract keypoints and state
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]

        # Use agent position (first two state dimensions)
        agent_pos = state[:, :2]

        # Flatten keypoints and concatenate with agent position
        obs = np.concatenate(
            [keypoint.reshape(keypoint.shape[0], -1), agent_pos], axis=-1
        )

        return {
            "obs": obs.astype(np.float32),  # (T, Do)
            "action": sample[self.action_key].astype(np.float32),  # (T, Da)
        }

    def get_normalizer(self, mode="limits", **kwargs) -> LinearNormalizer:
        # Fit a normalizer to the observation and action data
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])


# Dataset for PushT with visual (image) observations
class PushTImageDataset(BasePushTDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        # Keys are fixed: image, state, and action
        keys = ["img", "state", "action"]
        super().__init__(
            zarr_path=zarr_path,
            keys=keys,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
        )

    def _sample_to_data(self, sample):
        # Extract and normalize image data (transpose to channel-first)
        agent_pos = sample["state"][:, :2].astype(np.float32)
        image = np.moveaxis(sample["img"], -1, 1).astype(np.float32) / 255.0

        return {
            "obs": {
                "image": image,  # (T, 3, H, W)
                "agent_pos": agent_pos,  # (T, 2)
            },
            "action": sample["action"].astype(np.float32),  # (T, Da)
        }

    def get_normalizer(self, mode="limits", **kwargs) -> LinearNormalizer:
        # Normalize actions and agent positions; image normalizer handled separately
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :2],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # Add dummy normalizer for images if needed
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])
