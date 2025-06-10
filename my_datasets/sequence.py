from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from my_datasets.transformations.base import BaseTransform
from my_datasets.utils.replay_buffer import ReplayBuffer
from my_datasets.utils.sequence_sampler import SequenceSampler
from my_models.my_normalizers.base import BaseNormalizer
from utils import dict_apply


class SequenceDataset(Dataset):
    """
    A generic dataset for sampling sequences from an offline dataset.

    This class composes a ReplayBuffer, a SequenceSampler, and a Transform.
    This design is highly flexible and allows mixing and matching of data
    sources, sampling strategies, and data processing pipelines.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sampler: SequenceSampler,
        transform: BaseTransform,
    ):
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start_idx = self.sampler[idx]
        end_idx = start_idx + self.sampler.sequence_length

        raw_sample = {
            key: self.replay_buffer[key][start_idx:end_idx]
            for key in self.replay_buffer.root.keys()
            if key != "meta"
        }

        data = self.transform(raw_sample)
        print(data)
        # Convert all numpy arrays to torch tensors
        return dict_apply(data, torch.from_numpy)

    def get_normalizer(self) -> "BaseNormalizer":
        """Gets the appropriate normalizer from the transform."""
        return self.transform.get_normalizer(self.replay_buffer)
