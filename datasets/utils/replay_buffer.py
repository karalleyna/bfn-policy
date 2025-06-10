from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import zarr


class ReplayBuffer:
    """
    A generic replay buffer that stores and retrieves episode data from a
    Zarr directory. It supports both appending new episodes and reading
    existing data.
    """

    def __init__(self, zarr_store: zarr.Group):
        """
        Initializes the ReplayBuffer with an already open Zarr store.

        Args:
            zarr_store: An opened Zarr group object.
        """
        self.root = zarr_store

        if "meta" not in self.root:
            meta_group = self.root.create_group("meta")
        else:
            meta_group = self.root["meta"]

        if "episode_ends" not in meta_group:
            meta_group.create_dataset(
                "episode_ends", shape=(0,), dtype=np.int64, chunks=(1024,)
            )

    @classmethod
    def from_path(cls, path: Union[str, Path], mode: str = "a") -> "ReplayBuffer":
        """
        Creates or opens a Zarr-based replay buffer from a directory path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open(str(path), mode=mode)
        return cls(root)

    # --- FIX IS HERE: Use @property for dynamic calculation ---
    @property
    def episode_ends(self) -> np.ndarray:
        """Returns a numpy array of step indices where each episode ends."""
        return self.root["meta"]["episode_ends"][:]

    @property
    def n_episodes(self) -> int:
        """
        Returns the total number of episodes stored in the buffer.
        This is calculated dynamically to always be up-to-date.
        """
        return len(self.episode_ends)

    @property
    def n_steps(self) -> int:
        """
        Returns the total number of steps across all episodes.
        This is calculated dynamically to always be up-to-date.
        """
        if self.n_episodes == 0:
            return 0
        return self.episode_ends[-1]

    def get_episode_slice(self, episode_idx: int) -> slice:
        """
        Gets the slice object for a given episode index.
        """
        if not (0 <= episode_idx < self.n_episodes):
            raise IndexError(
                f"Episode index {episode_idx} is out of bounds for "
                f"a buffer with {self.n_episodes} episodes."
            )
        start = self.episode_ends[episode_idx - 1] if episode_idx > 0 else 0
        end = self.episode_ends[episode_idx]
        return slice(start, end)

    def __getitem__(self, key: str) -> zarr.Array:
        """Provides dictionary-style access to the data arrays."""
        return self.root[key]

    def add_episode(self, episode_data: Dict[str, np.ndarray]):
        """
        Adds a complete episode to the replay buffer.
        """
        if not episode_data:
            raise ValueError("Cannot add an empty episode.")

        episode_len = next(iter(episode_data.values())).shape[0]
        if not all(arr.shape[0] == episode_len for arr in episode_data.values()):
            raise ValueError("All data arrays in an episode must have the same length.")

        # `self.n_steps` is now a property, so it will fetch the current value
        start_idx = self.n_steps
        end_idx = start_idx + episode_len

        for key, value in episode_data.items():
            if key not in self.root:
                self.root.create_dataset(
                    key,
                    shape=(0,) + value.shape[1:],
                    chunks=(100,) + value.shape[1:],
                    dtype=value.dtype,
                )
            self.root[key].append(value)

        new_episode_ends = np.append(self.episode_ends, end_idx)
        self.root["meta"]["episode_ends"][:] = new_episode_ends

    def __repr__(self) -> str:
        """Provides a human-readable representation of the ReplayBuffer."""
        path_repr = "in-memory"
        if hasattr(self.root.store, "path"):
            path_repr = f"path='{self.root.store.path}'"

        data_keys = [key for key in self.root.keys() if key != "meta"]
        return (
            f"ReplayBuffer({path_repr}, n_episodes={self.n_episodes}, "
            f"n_steps={self.n_steps}, data_keys={data_keys})"
        )
