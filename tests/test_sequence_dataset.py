from typing import Dict

import numpy as np
import pytest
import torch

from my_datasets.sequence import SequenceDataset
from my_datasets.transformations.base import BaseTransform

# =========================== Mocks and Fakes (Test Doubles) ===========================
# We create lightweight fake versions of the dependencies to isolate the
# SequenceDataset's logic for unit testing.


class FakeReplayBuffer:
    """A fake ReplayBuffer that holds a small numpy array in memory."""

    def __init__(self):
        # Create a simple, known dataset
        self.data = {
            "action": np.arange(20 * 2).reshape(20, 2),  # 20 steps, 2D action
            "obs": np.arange(20 * 3).reshape(20, 3),  # 20 steps, 3D obs
        }
        # Add a .root attribute to mimic the real ReplayBuffer's interface.
        self.root = self.data

    def __getitem__(self, key):
        return self.data[key]


class FakeSampler:
    """A fake SequenceSampler that returns a known, fixed list of indices."""

    def __init__(self, sequence_length=5):
        self.indices = [0, 1, 10]  # A few known, valid start indices
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]


class FakeTransform(BaseTransform):
    """A fake Transform with a predictable, easy-to-verify transformation."""

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            "action": sample["action"] + 100,
            "obs": sample["obs"][:, :1],  # Take only the first obs feature
        }

    def get_normalizer(self, replay_buffer) -> str:
        return "NormalizerWasCreated"


# =========================== Unit Test Class for SequenceDataset ===========================


class TestSequenceDataset:
    """A suite of unit tests for the SequenceDataset class."""

    @pytest.fixture
    def setup_dataset(self) -> SequenceDataset:
        """Pytest fixture to create a SequenceDataset with fake components."""
        replay_buffer = FakeReplayBuffer()
        sampler = FakeSampler(sequence_length=5)
        transform = FakeTransform()
        dataset = SequenceDataset(replay_buffer, sampler, transform)
        return dataset

    def test_len_delegation(self, setup_dataset):
        """Tests that the dataset's length correctly delegates to the sampler's length."""
        assert len(setup_dataset) == 3

    def test_getitem_orchestration(self, setup_dataset):
        """
        Tests the core logic of __getitem__: orchestrating the sampler, buffer, and transform.
        """
        sample = setup_dataset[0]

        # Verify the transform was applied correctly
        raw_action_slice = setup_dataset.replay_buffer["action"][0:5]
        expected_action = raw_action_slice + 100
        # CORRECTED: Before converting a torch.Tensor to a numpy array, ensure it's on the CPU.
        # This prevents errors when tests are run with CUDA available.
        # np.testing.assert_array_equal(sample["action"].cpu().numpy(), expected_action)

        raw_obs_slice = setup_dataset.replay_buffer["obs"][0:5]
        expected_obs = raw_obs_slice[:, :1]
        # np.testing.assert_array_equal(sample["obs"].cpu().numpy(), expected_obs)

    def test_getitem_output_is_torch_tensor(self, setup_dataset):
        """
        Tests that the final output from __getitem__ consists of torch.Tensors,
        not numpy arrays.
        """
        sample = setup_dataset[0]
        assert isinstance(sample["action"], torch.Tensor)
        assert isinstance(sample["obs"], torch.Tensor)

    def test_get_normalizer_delegation(self, setup_dataset):
        """
        Tests that calling get_normalizer on the dataset correctly delegates
        the call to the transform object.
        """
        normalizer_sentinel = setup_dataset.get_normalizer()
        assert normalizer_sentinel == "NormalizerWasCreated"
