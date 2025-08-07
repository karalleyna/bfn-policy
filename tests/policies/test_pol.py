"""
Unit tests for the refactored policy framework.

This test suite verifies the functionality of the AbstractPolicy,
AbstractDiffusionPolicy, and VisionUnetDiffusionPolicy classes.
"""

from typing import Dict

import pytest
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from policies.base import AbstractPolicy
from policies.diffusion.multimodal_unet import VisionUnetDiffusionPolicy


class MockObsEncoder(nn.Module):
    """A mock vision encoder that returns a fixed-size feature vector."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        # A simple linear layer to simulate feature extraction
        self.layer = nn.Linear(3 * 64 * 64, output_dim)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Assuming 'rgb' is the key for image observations
        img_obs = obs_dict["rgb"]
        # Flatten the image before passing to the linear layer
        return self.layer(img_obs.flatten(start_dim=1))


class MockDenoisingNetwork(nn.Module):
    """
    A mock UNet-style denoising network that returns a tensor of the correct
    shape.
    """

    def __init__(self, trajectory_dim: int):
        super().__init__()
        self.layer = nn.Linear(trajectory_dim, trajectory_dim)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        # A simple pass-through to ensure the shape is maintained.
        # A real network would be much more complex.
        return self.layer(sample)


class MockSingleNormalizer:
    """A mock normalizer for a single data modality (e.g., action or obs)."""

    def normalize(self, data: Dict) -> Dict:
        # In a real scenario, this would apply mean/std normalization.
        # For testing, we just return the data as is.
        return data

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        # Identity function for testing.
        return data


# ====================================================================
# 2. Pytest Fixtures
# ====================================================================
@pytest.fixture
def config():
    """Provides a shared configuration dictionary for tests."""
    return {
        "horizon": 16,
        "action_dim": 6,
        "n_action_steps": 4,
        "n_obs_steps": 2,
        "obs_feature_dim": 128,
        "batch_size": 2,
    }


@pytest.fixture
def mock_noise_scheduler(config):
    """Provides a DDPMScheduler instance."""
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        beta_start=0.0001,
        beta_end=0.02,
    )


@pytest.fixture
def mock_normalizer():
    """Provides a mock normalizer dictionary."""
    return {"action": MockSingleNormalizer(), "obs": MockSingleNormalizer()}


@pytest.fixture
def policy_and_data(request, config, mock_noise_scheduler, mock_normalizer):
    """
    A comprehensive fixture to set up the policy and generate mock data.
    This fixture is parameterized by `obs_as_global_cond`.
    """
    obs_as_global_cond = request.param
    action_dim = config["action_dim"]
    obs_feature_dim = config["obs_feature_dim"]

    # Determine the input dimension for the denoising network based on the mode
    trajectory_dim = action_dim if obs_as_global_cond else action_dim + obs_feature_dim

    # Instantiate mock components
    denoising_net = MockDenoisingNetwork(trajectory_dim=trajectory_dim)
    obs_encoder = MockObsEncoder(output_dim=obs_feature_dim)

    # Instantiate the policy to be tested
    policy = VisionUnetDiffusionPolicy(
        denoising_network=denoising_net,
        obs_encoder=obs_encoder,
        shape_meta={},  # Not used in this test, can be empty
        horizon=config["horizon"],
        action_dim=action_dim,
        n_action_steps=config["n_action_steps"],
        n_obs_steps=config["n_obs_steps"],
        obs_as_global_cond=obs_as_global_cond,
        noise_scheduler=mock_noise_scheduler,
    )
    policy.set_normalizer(mock_normalizer)

    # Create mock data for testing
    B, T_obs, T_act = (
        config["batch_size"],
        config["n_obs_steps"],
        config["horizon"],
    )
    obs_dict = {"rgb": torch.randn(B, T_obs, 3, 64, 64)}
    batch = {
        "obs": {"rgb": torch.randn(B, T_act, 3, 64, 64)},
        "action": torch.randn(B, T_act, action_dim),
    }
    return policy, obs_dict, batch


# ====================================================================
# 3. Test Functions
# ====================================================================
def test_policy_initialization(config):
    """Tests that the policy can be instantiated without errors."""
    policy = VisionUnetDiffusionPolicy(
        denoising_network=nn.Linear(1, 1),
        obs_encoder=MockObsEncoder(config["obs_feature_dim"]),
        shape_meta={},
        horizon=config["horizon"],
        action_dim=config["action_dim"],
        n_action_steps=config["n_action_steps"],
        n_obs_steps=config["n_obs_steps"],
        noise_scheduler=DDPMScheduler(),
    )
    assert isinstance(policy, AbstractPolicy)
    assert policy.n_action_steps == config["n_action_steps"]


@pytest.mark.parametrize(
    "policy_and_data", [True, False], indirect=True, ids=["GlobalCond", "Inpainting"]
)
def test_predict_action(policy_and_data, config):
    """
    Tests the predict_action method for both global condition and inpainting modes.
    Verifies the output shape and type.
    """
    policy, obs_dict, _ = policy_and_data
    result = policy.predict_action(obs_dict)

    assert "action" in result
    assert "action_pred" in result

    # Check the shape of the executable action
    expected_action_shape = (
        config["batch_size"],
        config["n_action_steps"],
        config["action_dim"],
    )
    assert result["action"].shape == expected_action_shape

    # Check the shape of the full predicted action trajectory
    expected_action_pred_shape = (
        config["batch_size"],
        config["horizon"],
        config["action_dim"],
    )
    assert result["action_pred"].shape == expected_action_pred_shape
    assert result["action"].dtype == torch.float32


@pytest.mark.parametrize(
    "policy_and_data", [True, False], indirect=True, ids=["GlobalCond", "Inpainting"]
)
def test_compute_loss(policy_and_data):
    """
    Tests the compute_loss method for both global condition and inpainting modes.
    Verifies that it returns a scalar loss.
    """
    policy, _, batch = policy_and_data
    loss = policy.compute_loss(batch)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Should be a scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
