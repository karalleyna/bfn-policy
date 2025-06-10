import numpy as np

from my_datasets.utils.replay_buffer import ReplayBuffer


class SequenceSampler:
    """
    Calculates all possible start indices for sampling fixed-length sequences
    from a ReplayBuffer, while respecting episode boundaries. This class
    generates the "recipe" for sampling but does not load data itself.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        episode_mask: np.ndarray = None,
    ):
        """
        Initializes the sampler.

        Args:
            replay_buffer: The ReplayBuffer to sample from.
            sequence_length: The length of the sequences to sample.
            pad_before: Number of steps to pad before the start of an episode.
            pad_after: Number of steps to pad after the end of an episode.
            episode_mask: A boolean array to select which episodes to sample from.
        """
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.indices = []
        n_episodes = replay_buffer.n_episodes
        if episode_mask is None:
            episode_mask = np.ones(n_episodes, dtype=bool)

        for i in range(n_episodes):
            if not episode_mask[i]:
                continue

            episode_slice = self.replay_buffer.get_episode_slice(i)
            min_start_idx = episode_slice.start - self.pad_before
            max_start_idx = episode_slice.stop + self.pad_after - self.sequence_length

            # Generate all valid start indices for this episode
            self.indices.extend(range(min_start_idx, max_start_idx + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> int:
        return self.indices[idx]
