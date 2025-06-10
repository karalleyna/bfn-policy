from typing import Tuple

from my_datasets.sequence import SequenceDataset
from my_datasets.utils.replay_buffer import ReplayBuffer
from my_datasets.utils.sequence_sampler import SequenceSampler


def create_pusht_dataset(
    zarr_path: str,
    obs_modality: str,
    horizon: int,
    pad_before: int = 0,
    pad_after: int = 0,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[SequenceDataset, SequenceDataset]:
    """
    Factory function to create train and validation datasets.
    This encapsulates the logic of splitting data and assembling components.
    """
    replay_buffer = ReplayBuffer.from_path(zarr_path)

    if obs_modality == "low_dim":
        transform = PushTLowDimTransform()
    elif obs_modality == "image":
        transform = PushTImageTransform()
    else:
        raise ValueError(f"Unknown modality: {obs_modality}")

    val_mask = np.zeros(replay_buffer.n_episodes, dtype=bool)
    val_indices = np.random.RandomState(seed).choice(
        replay_buffer.n_episodes,
        size=int(replay_buffer.n_episodes * val_ratio),
        replace=False,
    )
    val_mask[val_indices] = True
    train_mask = ~val_mask

    train_sampler = SequenceSampler(
        replay_buffer, horizon, pad_before, pad_after, train_mask
    )
    val_sampler = SequenceSampler(
        replay_buffer, horizon, pad_before, pad_after, val_mask
    )

    train_dataset = SequenceDataset(replay_buffer, train_sampler, transform)
    val_dataset = SequenceDataset(replay_buffer, val_sampler, transform)

    return train_dataset, val_dataset
