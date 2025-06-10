import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

# Assuming the ReplayBuffer class is in a file named `my_dataset/replay_buffer.py`
from my_datasets.utils.replay_buffer import ReplayBuffer

# =========================== Test Fixtures (Reusable Setups) ===========================


@pytest.fixture
def temp_dir_path() -> Path:
    """Creates a temporary directory and cleans it up after the test."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def populated_zarr_path(temp_dir_path: Path) -> str:
    """
    Creates a temporary, populated Zarr dataset on disk for testing read operations.
    """
    zarr_path = temp_dir_path / "populated_dataset.zarr"
    root = zarr.open(str(zarr_path), "w")

    # --- FIX IS HERE: Use the 'data' keyword argument, not 'fill_value' ---
    # The 'data' argument is used to write an entire existing array.
    # The 'fill_value' argument is only for a single scalar value.
    root.create_array(
        "action", shape=(23, 2), fill_value=np.arange(23 * 2).reshape(23, 2), dtype="f4"
    )
    root.create_array(
        "state", shape=(23, 4), fill_value=np.arange(23 * 4).reshape(23, 4), dtype="f4"
    )

    # Create metadata with known episode boundaries
    meta = root.create_group("meta")
    # Also correcting the DeprecationWarning by using create_array for consistency
    meta.create_array("episode_ends", shape=(3,), fill_value=np.array([10, 15, 23]))
    return str(zarr_path)


# =========================== Unit Test Class for ReplayBuffer ===========================


class TestReplayBuffer:
    """A suite of unit tests for the ReplayBuffer class."""

    # --- Tests for Reading from an Existing Buffer ---

    def test_from_path_instantiation(self, populated_zarr_path):
        """
        Tests that the ReplayBuffer can be successfully instantiated from a file path.
        """
        replay_buffer = ReplayBuffer.from_path(populated_zarr_path, mode="r")
        assert replay_buffer is not None
        assert isinstance(replay_buffer.root, zarr.Group)

    def test_properties_are_correct(self, populated_zarr_path):
        """
        Tests that the `n_episodes` and `n_steps` properties are read correctly.
        """
        replay_buffer = ReplayBuffer.from_path(populated_zarr_path, mode="r")
        assert replay_buffer.n_episodes == 3
        assert replay_buffer.n_steps == 23

    def test_getitem_access(self, populated_zarr_path):
        """
        Tests dictionary-style access to data arrays and verifies their content.
        """
        replay_buffer = ReplayBuffer.from_path(populated_zarr_path, mode="r")
        action_data = replay_buffer["action"]
        assert isinstance(action_data, zarr.Array)
        assert action_data.shape == (23, 2)
        expected_action_data = np.arange(23 * 2).reshape(23, 2)
        np.testing.assert_array_equal(action_data[:], expected_action_data)

    def test_get_episode_slice_logic(self, populated_zarr_path):
        """
        Tests that `get_episode_slice` returns the correct start/end indices for
        various episodes.
        """
        replay_buffer = ReplayBuffer.from_path(populated_zarr_path, mode="r")
        assert replay_buffer.get_episode_slice(0) == slice(0, 10)
        assert replay_buffer.get_episode_slice(1) == slice(10, 15)
        assert replay_buffer.get_episode_slice(2) == slice(15, 23)

    def test_get_episode_slice_out_of_bounds(self, populated_zarr_path):
        """Tests that accessing an invalid episode index raises an IndexError."""
        replay_buffer = ReplayBuffer.from_path(populated_zarr_path, mode="r")
        with pytest.raises(IndexError):
            replay_buffer.get_episode_slice(3)
        with pytest.raises(IndexError):
            replay_buffer.get_episode_slice(-1)

    # --- Tests for Creating and Writing to a New Buffer ---

    def test_initialization_on_empty_path(self, temp_dir_path):
        """
        Tests that creating a ReplayBuffer with a new path results in a
        correctly structured empty buffer. This directly tests the new logic
        in the __init__ method.
        """
        empty_path = temp_dir_path / "new_buffer.zarr"
        replay_buffer = ReplayBuffer.from_path(empty_path)

        assert "meta" in replay_buffer.root
        assert "episode_ends" in replay_buffer.root["meta"]
        assert replay_buffer.n_episodes == 0
        assert replay_buffer.n_steps == 0

    def test_add_first_episode(self, temp_dir_path):
        """Tests adding the very first episode to an empty buffer."""
        empty_path = temp_dir_path / "new_buffer.zarr"
        replay_buffer = ReplayBuffer.from_path(empty_path)

        episode_data = {
            "action": np.random.rand(50, 2).astype(np.float32),
            "obs": np.random.rand(50, 8).astype(np.float32),
        }
        replay_buffer.add_episode(episode_data)

        assert replay_buffer.n_episodes == 1
        assert replay_buffer.n_steps == 50
        np.testing.assert_array_equal(replay_buffer.episode_ends, np.array([50]))
        assert replay_buffer["action"].shape == (50, 2)

    def test_add_subsequent_episodes(self, temp_dir_path):
        """Tests adding multiple episodes and verifies the metadata."""
        path = temp_dir_path / "multi_ep_buffer.zarr"
        replay_buffer = ReplayBuffer.from_path(path)

        replay_buffer.add_episode({"action": np.zeros((50, 2), dtype=np.float32)})
        assert replay_buffer.n_steps == 50
        assert replay_buffer.n_episodes == 1

        replay_buffer.add_episode({"action": np.zeros((30, 2), dtype=np.float32)})
        assert replay_buffer.n_steps == 80  # 50 + 30
        assert replay_buffer.n_episodes == 2
        np.testing.assert_array_equal(replay_buffer.episode_ends, np.array([50, 80]))

    def test_add_episode_with_mismatched_lengths_error(self, temp_dir_path):
        """Tests that adding an episode with inconsistent lengths raises a ValueError."""
        path = temp_dir_path / "bad_ep_buffer.zarr"
        replay_buffer = ReplayBuffer.from_path(path)

        episode_data = {
            "action": np.zeros((10, 2)),  # Length 10
            "obs": np.zeros((11, 8)),  # Length 11
        }
        with pytest.raises(
            ValueError, match="All data arrays in an episode must have the same length"
        ):
            replay_buffer.add_episode(episode_data)
