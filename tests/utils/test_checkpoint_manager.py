"""Unit tests for the TopKCheckpointManager."""

import os
from unittest.mock import patch

import pytest

# Assuming the manager is in utils/checkpoint_manager.py
from utils.checkpoint_manager import TopKCheckpointManager


@pytest.fixture
def manager_min_mode(tmp_path):
    """Fixture for a TopKCheckpointManager in 'min' mode."""
    return TopKCheckpointManager(
        save_dir=str(tmp_path),
        monitor_key="val_loss",
        mode="min",
        k=2,
        format_str="epoch={epoch:02d}-loss={val_loss:.2f}.ckpt",
    )


@pytest.fixture
def manager_max_mode(tmp_path):
    """Fixture for a TopKCheckpointManager in 'max' mode."""
    return TopKCheckpointManager(
        save_dir=str(tmp_path),
        monitor_key="accuracy",
        mode="max",
        k=2,
        format_str="epoch={epoch:02d}-acc={accuracy:.2f}.ckpt",
    )


def test_initialization():
    """Tests that the manager initializes correctly."""
    with pytest.raises(ValueError):
        TopKCheckpointManager(save_dir=".", monitor_key="k", mode="invalid")
    with pytest.raises(ValueError):
        TopKCheckpointManager(save_dir=".", monitor_key="k", k=-1)


def test_k_zero_returns_none(manager_min_mode):
    """Tests that no path is returned when k=0."""
    manager_min_mode.k = 0
    path = manager_min_mode.on_validation_end({"epoch": 1, "val_loss": 0.5})
    assert path is None


def test_monitor_key_missing(manager_min_mode, caplog):
    """Tests that a warning is logged if the monitor key is missing."""
    path = manager_min_mode.on_validation_end({"epoch": 1, "other_metric": 0.5})
    assert path is None
    assert "Monitor key 'val_loss' not found" in caplog.text


def test_fill_capacity_min_mode(manager_min_mode, tmp_path):
    """Tests filling the checkpoint capacity in 'min' mode."""
    # First checkpoint
    metrics1 = {"epoch": 1, "val_loss": 0.8}
    path1 = manager_min_mode.on_validation_end(metrics1)
    expected_path1 = tmp_path / "epoch=01-loss=0.80.ckpt"
    assert path1 == str(expected_path1)
    assert manager_min_mode.best_checkpoint_path == str(expected_path1)

    # Second checkpoint (better)
    metrics2 = {"epoch": 2, "val_loss": 0.6}
    path2 = manager_min_mode.on_validation_end(metrics2)
    expected_path2 = tmp_path / "epoch=02-loss=0.60.ckpt"
    assert path2 == str(expected_path2)
    assert manager_min_mode.best_checkpoint_path == str(expected_path2)

    # Heap should contain both
    assert len(manager_min_mode._heap) == 2


def test_min_mode_logic(manager_min_mode, tmp_path):
    """Tests replacement logic in 'min' mode."""
    # Pre-fill
    manager_min_mode.on_validation_end({"epoch": 1, "val_loss": 0.8})
    manager_min_mode.on_validation_end({"epoch": 2, "val_loss": 0.6})

    # This one is worse, should not be saved
    path_worse = manager_min_mode.on_validation_end({"epoch": 3, "val_loss": 0.9})
    assert path_worse is None
    assert len(manager_min_mode._heap) == 2

    # This one is better, should replace the worst (0.8)
    metrics_better = {"epoch": 4, "val_loss": 0.5}
    path_better = manager_min_mode.on_validation_end(metrics_better)
    expected_path = tmp_path / "epoch=04-loss=0.50.ckpt"
    assert path_better == str(expected_path)
    assert len(manager_min_mode._heap) == 2
    assert manager_min_mode.best_checkpoint_path == str(expected_path)


def test_max_mode_logic(manager_max_mode, tmp_path):
    """Tests replacement logic in 'max' mode."""
    # Pre-fill
    manager_max_mode.on_validation_end({"epoch": 1, "accuracy": 0.90})
    manager_max_mode.on_validation_end({"epoch": 2, "accuracy": 0.92})

    # This one is worse, should not be saved
    path_worse = manager_max_mode.on_validation_end({"epoch": 3, "accuracy": 0.88})
    assert path_worse is None
    assert len(manager_max_mode._heap) == 2

    # This one is better, should replace the worst (0.90)
    metrics_better = {"epoch": 4, "accuracy": 0.95}
    path_better = manager_max_mode.on_validation_end(metrics_better)
    expected_path = tmp_path / "epoch=04-acc=0.95.ckpt"
    assert path_better == str(expected_path)
    assert len(manager_max_mode._heap) == 2
    # In max mode, the best value is the largest, but it's stored as the most negative
    # so the best checkpoint path is still at the root of the min-heap.
    assert manager_max_mode.best_checkpoint_path == str(expected_path)


@patch("os.remove")
def test_file_removal(mock_remove, manager_min_mode, tmp_path):
    """Tests that old checkpoint files are correctly removed."""
    # Path of the first (and worst) checkpoint
    worst_path = tmp_path / "epoch=01-loss=0.80.ckpt"
    # Create a dummy file to be removed.
    worst_path.touch()
    assert worst_path.exists()

    # Pre-fill the manager
    manager_min_mode.on_validation_end({"epoch": 1, "val_loss": 0.8})
    manager_min_mode.on_validation_end({"epoch": 2, "val_loss": 0.6})

    # This new checkpoint should cause the removal of the worst one.
    manager_min_mode.on_validation_end({"epoch": 3, "val_loss": 0.5})

    # Check that os.remove was called with the correct path
    mock_remove.assert_called_once_with(str(worst_path))
