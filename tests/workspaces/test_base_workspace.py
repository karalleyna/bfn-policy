"""Pytest-based unit tests for the BaseWorkspace class."""

import logging
import pathlib
import tempfile
from typing import Any, Dict, Generator, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Import the class to be tested
# In a real project, this would be: from my_project.workspaces import BaseWorkspace
from workspaces.base import BaseWorkspace

# Disable most logging to keep test output clean, but can be changed for debugging.
logging.basicConfig(level=logging.CRITICAL)


# --- Helper Functions and Classes (unchanged from unittest version) ---


def compare_state_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> bool:
    """Recursively compares two state dictionaries for equality."""
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        item1 = dict1[key]
        item2 = dict2[key]
        if isinstance(item1, torch.Tensor):
            if not torch.equal(item1, item2):
                return False
        elif isinstance(item1, dict):
            if not compare_state_dicts(item1, item2):
                return False
        elif item1 != item2:
            return False
    return True


class SimpleModel(nn.Module):
    """A minimal model for testing state dict saving and loading."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class TestWorkspace(BaseWorkspace):
    """A concrete implementation of BaseWorkspace for testing purposes."""

    include_keys = ("epoch_count",)
    exclude_keys = ("sensitive_data",)

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir)
        self.model = SimpleModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.epoch_count = 0
        self.some_metric = 0.99
        self.sensitive_data = "do_not_save_this"


# --- Pytest Fixture ---


@pytest.fixture
def workspace_setup() -> Generator[Tuple[pathlib.Path, OmegaConf], None, None]:
    """
    Pytest fixture to set up a temporary directory and mock config for each test.

    This replaces the setUp and tearDown methods from the unittest.TestCase.

    Yields:
        A tuple containing the path to the temporary output directory and a
        mock OmegaConf object.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = pathlib.Path(temp_dir)
        config = OmegaConf.create(
            {"model": {"name": "SimpleModel"}, "optimizer": {"lr": 0.01}}
        )
        yield output_path, config
    # The temporary directory is automatically cleaned up by the context manager.


# --- Test Functions ---


def test_initialization_and_output_dir(workspace_setup):
    """
    Tests that the workspace initializes correctly and resolves the output_dir.
    """
    output_path, config = workspace_setup
    workspace = TestWorkspace(config, output_dir=str(output_path))

    assert isinstance(workspace, BaseWorkspace)
    assert workspace.output_dir == output_path
    assert workspace.epoch_count == 0
    assert "model" in workspace.__dict__
    assert "optimizer" in workspace.__dict__


def test_save_and_load_checkpoint_sync(workspace_setup):
    """Tests the core checkpointing logic in synchronous mode."""
    output_path, cfg = workspace_setup
    # --- 1. Create and configure the initial workspace ---
    workspace1 = TestWorkspace(cfg, output_dir=str(output_path))
    workspace1.epoch_count = 10
    workspace1.optimizer.step()

    # --- 2. Save the checkpoint ---
    checkpoint_path = workspace1.save_checkpoint(tag="epoch_10", use_thread=False)
    assert pathlib.Path(checkpoint_path).exists()

    # --- 3. Create a new, clean workspace and load the state ---
    workspace2 = TestWorkspace(cfg, output_dir=str(output_path))
    workspace2.load_checkpoint(path=checkpoint_path)

    # --- 4. Assert that the state was restored correctly ---
    assert workspace1.epoch_count == workspace2.epoch_count
    assert compare_state_dicts(
        workspace1.model.state_dict(), workspace2.model.state_dict()
    )
    assert compare_state_dicts(
        workspace1.optimizer.state_dict(), workspace2.optimizer.state_dict()
    )
    # Check that a non-included attribute was NOT restored.
    assert workspace1.some_metric != workspace2.some_metric


def test_save_and_load_checkpoint_async(workspace_setup):
    """Tests the asynchronous (threaded) checkpointing logic."""
    output_path, cfg = workspace_setup
    workspace1 = TestWorkspace(cfg, output_dir=str(output_path))
    workspace1.epoch_count = 5
    workspace1.optimizer.step()

    checkpoint_path = workspace1.save_checkpoint(tag="async_test", use_thread=True)
    workspace1.wait_for_saving()  # Block until the background thread is finished.
    assert pathlib.Path(checkpoint_path).exists()

    workspace2 = TestWorkspace(cfg, output_dir=str(output_path))
    workspace2.load_checkpoint(path=checkpoint_path)

    assert 5 == workspace2.epoch_count
    assert compare_state_dicts(
        workspace1.model.state_dict(), workspace2.model.state_dict()
    )


def test_create_from_checkpoint(workspace_setup):
    """Tests the factory method for creating an instance from a checkpoint."""
    output_path, cfg = workspace_setup
    workspace1 = TestWorkspace(cfg, output_dir=str(output_path))
    workspace1.epoch_count = 42
    workspace1.optimizer.step()
    checkpoint_path = workspace1.save_checkpoint(use_thread=False)

    # Create a new instance directly from the file.
    workspace2 = TestWorkspace.create_from_checkpoint(path=checkpoint_path)

    assert isinstance(workspace2, TestWorkspace)
    assert workspace1.epoch_count == workspace2.epoch_count
    assert OmegaConf.select(workspace2.cfg, "model.name") == "SimpleModel"
    assert compare_state_dicts(
        workspace1.model.state_dict(), workspace2.model.state_dict()
    )


def test_include_and_exclude_keys(workspace_setup):
    """Tests that the include_keys and exclude_keys logic works correctly."""
    output_path, cfg = workspace_setup
    workspace1 = TestWorkspace(cfg, output_dir=str(output_path))
    workspace1.epoch_count = 100
    workspace1.sensitive_data = "should_not_be_saved"
    workspace1.some_metric = 3.14

    # Test with custom keys provided at save time.
    checkpoint_path = workspace1.save_checkpoint(
        tag="key_test",
        include_keys=("some_metric",),  # Override class default
        exclude_keys=("model",),  # Exclude the model
        use_thread=False,
    )

    workspace2 = TestWorkspace.create_from_checkpoint(path=checkpoint_path)

    assert workspace2.some_metric == pytest.approx(3.14)
    assert workspace2.epoch_count == 0  # Was not included in this specific save.
    assert not compare_state_dicts(  # Model was excluded.
        workspace1.model.state_dict(), workspace2.model.state_dict()
    )


def test_snapshot_functionality(workspace_setup):
    """Tests the less-robust snapshot saving/loading mechanism."""
    output_path, cfg = workspace_setup
    workspace1 = TestWorkspace(cfg, output_dir=str(output_path))
    workspace1.epoch_count = 77
    snapshot_path = workspace1.save_snapshot()
    assert pathlib.Path(snapshot_path).exists()

    workspace2 = TestWorkspace.create_from_snapshot(snapshot_path)

    assert isinstance(workspace2, TestWorkspace)
    assert workspace1.epoch_count == workspace2.epoch_count
    assert workspace1.some_metric == pytest.approx(workspace2.some_metric)


def test_error_handling_for_missing_file(workspace_setup):
    """Tests that an appropriate error is raised for missing files."""
    output_path, cfg = workspace_setup
    workspace = TestWorkspace(cfg, output_dir=str(output_path))
    missing_path = output_path / "non_existent.ckpt"

    with pytest.raises(FileNotFoundError):
        workspace.load_checkpoint(path=str(missing_path))

    with pytest.raises(FileNotFoundError):
        TestWorkspace.create_from_snapshot(str(missing_path))
