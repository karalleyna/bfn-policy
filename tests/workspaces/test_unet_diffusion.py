"""
Unit tests for the DiffusionUNetHybridWorkspace.

This test suite uses mocks to isolate the workspace from external dependencies
such as data loaders, loggers, and environment runners, allowing for focused
testing of the workspace's internal logic.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

# This import path needs to be correct relative to the project root
from workspaces.diffusion_unet_hybrid import DiffusionUNetHybridWorkspace, dict_apply


@pytest.fixture
def minimal_cfg():
    """Provides a minimal, valid Hydra configuration for testing."""
    return OmegaConf.create(
        {
            "training": {
                "seed": 42,
                "use_ema": True,
                "resume": False,
                "device": "cpu",
                "debug": False,
                "gradient_accumulate_every": 1,
                "tqdm_interval_sec": 10,
                "num_epochs": 1,
                "max_train_steps": 10,  # FIX: Set higher than loader length to ensure loop completes
                "max_val_steps": 2,
                "rollout_every": 1,
                "checkpoint_every": 1,
                "val_every": 1,
                "sample_every": 1,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
            },
            "policy": {"_target_": "unittest.mock.MagicMock"},
            "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-4},
            "ema": {"_target_": "unittest.mock.MagicMock"},
            "task": {
                "dataset": {"_target_": "unittest.mock.MagicMock"},
                "env_runner": {"_target_": "unittest.mock.MagicMock"},
            },
            "dataloader": {"batch_size": 2},
            "val_dataloader": {"batch_size": 2},
            "logging": {"name": "test_run"},
            "checkpoint": {
                "topk": {"monitor_key": "val_loss", "mode": "min", "k": 1},
                "save_last_ckpt": True,
                "save_last_snapshot": True,
            },
        }
    )


@patch("workspaces.diffusion_unet_hybrid.hydra.utils.instantiate")
def test_workspace_initialization(mock_instantiate, minimal_cfg):
    """Tests that the workspace and its components are initialized correctly."""
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    mock_optimizer = torch.optim.AdamW(mock_model.parameters())

    def instantiate_side_effect(cfg, **kwargs):
        if "params" in kwargs:
            return mock_optimizer
        return mock_model

    mock_instantiate.side_effect = instantiate_side_effect

    workspace = DiffusionUNetHybridWorkspace(minimal_cfg)

    assert isinstance(workspace, DiffusionUNetHybridWorkspace)
    assert workspace.model is not None
    assert workspace.optimizer is mock_optimizer


@patch("workspaces.diffusion_unet_hybrid.wandb")
@patch("workspaces.diffusion_unet_hybrid.DataLoader")
@patch("workspaces.diffusion_unet_hybrid.TopKCheckpointManager")
@patch("workspaces.diffusion_unet_hybrid.JsonLogger")
@patch("workspaces.diffusion_unet_hybrid.get_scheduler")
@patch("workspaces.diffusion_unet_hybrid.hydra.utils.instantiate")
@patch("workspaces.diffusion_unet_hybrid.optimizer_to")
@patch("workspaces.diffusion_unet_hybrid.BaseWorkspace.save_checkpoint")
@patch("workspaces.diffusion_unet_hybrid.BaseWorkspace.save_snapshot")
def test_run_method(
    mock_save_snapshot,
    mock_save_checkpoint,
    mock_optimizer_to,
    mock_instantiate,
    mock_get_scheduler,
    mock_json_logger,
    mock_topk_manager,
    mock_dataloader,
    mock_wandb,
    minimal_cfg,
    tmp_path,
):
    """
    Tests that the main `run` method executes the training loop without errors.
    """
    # --- Mock Instantiated Objects ---
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    mock_model.compute_loss.return_value = torch.tensor(0.5, requires_grad=True)
    mock_model.predict_action.return_value = {"action_pred": torch.zeros(1)}

    mock_optimizer = torch.optim.AdamW(mock_model.parameters())
    mock_optimizer.step = MagicMock()
    mock_optimizer.zero_grad = MagicMock()

    mock_dataset = MagicMock()
    mock_dataset.get_normalizer.return_value = MagicMock()
    mock_dataset.get_validation_dataset.return_value = MagicMock()

    def instantiate_side_effect(cfg, **kwargs):
        if "params" in kwargs:
            return mock_optimizer
        if (
            cfg.get("task", {}).get("dataset", {}).get("_target_")
            == "unittest.mock.MagicMock"
        ):
            return mock_dataset
        return mock_model

    mock_instantiate.side_effect = instantiate_side_effect

    mock_scheduler = MagicMock()
    mock_scheduler.get_last_lr.return_value = [1e-4]
    mock_get_scheduler.return_value = mock_scheduler

    # --- Mock Logger Context Manager ---
    # FIX: Explicitly mock the object returned by the context manager's __enter__
    mock_logger_instance = MagicMock()
    mock_json_logger.return_value.__enter__.return_value = mock_logger_instance

    # --- Mock DataLoaders ---
    dummy_batch = {"obs": torch.zeros(1), "action": torch.zeros(1)}
    mock_train_loader = MagicMock()
    mock_train_loader.__len__.return_value = 2
    mock_train_loader.__iter__.return_value = iter([dummy_batch, dummy_batch])

    mock_val_loader = MagicMock()
    mock_val_loader.__iter__.return_value = iter([dummy_batch])
    mock_dataloader.side_effect = [mock_train_loader, mock_val_loader]

    # --- Instantiate and Run ---
    workspace = DiffusionUNetHybridWorkspace(minimal_cfg, output_dir=str(tmp_path))

    workspace.optimizer = mock_optimizer
    workspace.model = mock_model
    workspace.ema_model = mock_model
    workspace.include_keys = tuple(workspace.include_keys)

    workspace.run()

    assert workspace.model.compute_loss.called
    assert mock_optimizer.step.called
    assert mock_topk_manager.return_value.get_ckpt_path.called
    assert mock_logger_instance.log.called
    assert mock_wandb.init.called
    assert mock_get_scheduler.called
    assert mock_save_checkpoint.called
    assert mock_save_snapshot.called
    assert workspace.global_step > 0
    assert workspace.epoch > 0
