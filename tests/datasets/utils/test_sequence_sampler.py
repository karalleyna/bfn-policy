import numpy as np
import pytest
import zarr

from datasets.utils.replay_buffer import ReplayBuffer
from datasets.utils.sequence_sampler import SequenceSampler


@pytest.fixture
def mock_replay_buffer() -> ReplayBuffer:
    """
    Provides a mock ReplayBuffer with a known, simple structure for testing.
    - Episode 0: 10 steps (indices 0-9)
    - Episode 1: 5 steps (indices 10-14)
    - Episode 2: 8 steps (indices 15-22)
    """
    store = zarr.storage.MemoryStore()
    root = zarr.group(store=store)

    root.create_dataset("action", shape=(23,), dtype="f4")

    meta = root.create_group("meta")

    meta.create_dataset("episode_ends", shape=(3,), data=np.array([10, 15, 23]))

    return ReplayBuffer(root)


class TestSequenceSampler:
    """A suite of unit tests for the SequenceSampler class."""

    def test_basic_sampling_no_padding(self, mock_replay_buffer):
        """
        Tests the sampler's core logic without any padding.
        It should generate all possible start indices within episode boundaries.
        """
        sampler = SequenceSampler(replay_buffer=mock_replay_buffer, sequence_length=4)

        # Expected indices:
        # Ep 0 (len 10): 0, 1, 2, 3, 4, 5, 6  (7 indices)
        # Ep 1 (len 5): 10, 11               (2 indices)
        # Ep 2 (len 8): 15, 16, 17, 18, 19   (5 indices)
        # Total: 7 + 2 + 5 = 14

        expected_len = 14
        assert len(sampler) == expected_len

        expected_indices = list(range(7)) + [10, 11] + list(range(15, 20))
        assert sampler.indices == expected_indices

    def test_padding_before(self, mock_replay_buffer):
        """
        Tests that `pad_before` correctly allows sampling to start "before" an episode.
        """
        sampler = SequenceSampler(
            replay_buffer=mock_replay_buffer, sequence_length=4, pad_before=2
        )

        # Expected indices now include starts from padding:
        # Ep 0: -2, -1, 0, 1, 2, 3, 4, 5, 6  (9 indices)
        # Ep 1: 8, 9, 10, 11                (4 indices)
        # Ep 2: 13, 14, 15, 16, 17, 18, 19  (7 indices)
        # Total: 9 + 4 + 7 = 20

        expected_len = 20
        assert len(sampler) == expected_len

        # Check a specific boundary case
        assert sampler.indices[0] == -2  # First possible start for ep 0
        assert sampler.indices[9] == 8  # First possible start for ep 1

    def test_padding_after(self, mock_replay_buffer):
        """
        Tests that `pad_after` correctly allows sampling to "end" after an episode.
        """
        sampler = SequenceSampler(
            replay_buffer=mock_replay_buffer, sequence_length=4, pad_after=2
        )

        # Expected indices now include starts that go past the episode end:
        # Ep 0 (ends at 10): starts up to 10+2-4=8. Indices 0-8 (9 indices)
        # Ep 1 (ends at 15): starts up to 15+2-4=13. Indices 10-13 (4 indices)
        # Ep 2 (ends at 23): starts up to 23+2-4=21. Indices 15-21 (7 indices)
        # Total: 9 + 4 + 7 = 20

        expected_len = 20
        assert len(sampler) == expected_len

        expected_indices = list(range(9)) + list(range(10, 14)) + list(range(15, 22))
        assert sampler.indices == expected_indices

    def test_episode_masking(self, mock_replay_buffer):
        """
        Tests that the `episode_mask` correctly filters which episodes are sampled from.
        """
        # Create a mask to only sample from Episode 0 and Episode 2
        episode_mask = np.array([True, False, True])

        sampler = SequenceSampler(
            replay_buffer=mock_replay_buffer,
            sequence_length=4,
            episode_mask=episode_mask,
        )

        # Expected indices:
        # Ep 0 (len 10): 0, 1, 2, 3, 4, 5, 6  (7 indices)
        # Ep 1 (masked out):                  (0 indices)
        # Ep 2 (len 8): 15, 16, 17, 18, 19   (5 indices)
        # Total: 7 + 5 = 12

        expected_len = 12
        assert len(sampler) == expected_len

        expected_indices = list(range(7)) + list(range(15, 20))
        assert sampler.indices == expected_indices

    def test_edge_case_sequence_equals_episode_length(self, mock_replay_buffer):
        """
        Tests that if sequence length equals episode length, exactly one sample is generated.
        """
        sampler = SequenceSampler(
            replay_buffer=mock_replay_buffer,
            sequence_length=5,  # Same length as Episode 1
        )

        # For episode 1 (indices 10-14, len 5), the only valid start is 10.
        # Check if 10 is in the generated indices.
        assert 10 in sampler.indices

        # Check that 11 is NOT a valid start index for a sequence of length 5.
        assert 11 not in sampler.indices

    def test_edge_case_sequence_gt_episode_length(self, mock_replay_buffer):
        """
        Tests that if sequence length is greater than episode length, no samples are generated.
        """
        sampler = SequenceSampler(
            replay_buffer=mock_replay_buffer, sequence_length=6  # Longer than Episode 1
        )

        # No valid start index should exist for episode 1 (indices 10-14).
        # Therefore, indices should jump from ep 0 to ep 2.
        # Max start for ep 0 (len 10) is 10-6=4. So indices are 0,1,2,3,4.
        # Min start for ep 2 (idx 15) is 15.
        assert 4 in sampler.indices
        assert 15 in sampler.indices
        assert 10 not in sampler.indices  # 10 is start of ep 1, should be skipped.
