import pytest
import torch
from torch import nn

from models.ema import ExponentialMovingAverage

# =========================== Test Fixtures (Reusable Setups) ===========================


@pytest.fixture
def simple_model() -> nn.Module:
    """Provides a simple, consistent model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Use buffer for non-trainable state and parameter for trainable state
            self.layer = nn.Linear(10, 2)
            self.register_buffer("running_mean", torch.zeros(10))

    return SimpleModel()


# =========================== Unit Test Class ===========================


class TestExponentialMovingAverage:
    """A suite of unit tests for the ExponentialMovingAverage class."""

    def test_initialization(self, simple_model):
        """Tests that the EMA initializes correctly with a shadow model and zero steps."""
        ema = ExponentialMovingAverage(simple_model, decay=0.9)
        assert ema.optimization_step == 0

        # Check that the EMA model is a separate object with the same initial parameters
        assert ema.ema_model is not simple_model
        torch.testing.assert_close(
            ema.ema_model.state_dict()["layer.weight"],
            simple_model.state_dict()["layer.weight"],
        )

    def test_decay_calculation_no_warmup(self, simple_model):
        """Tests that decay is constant when warmup is disabled."""
        decay_rate = 0.95
        ema = ExponentialMovingAverage(simple_model, decay=decay_rate, use_warmup=False)

        # Decay should always be the fixed value
        assert ema._get_current_decay() == decay_rate
        ema.optimization_step += 100
        assert ema._get_current_decay() == decay_rate

    def test_decay_calculation_with_warmup(self, simple_model):
        """Tests the warmup schedule for the decay rate."""
        ema = ExponentialMovingAverage(simple_model, decay=0.999, update_after_step=10)

        # Before update_after_step, decay should be 0.0
        ema.optimization_step = torch.tensor(5)
        assert ema._get_current_decay() == 0.0

        # At update_after_step, decay should still be 0.0 (step - update_after_step - 1 < 0)
        ema.optimization_step = torch.tensor(10)
        assert ema._get_current_decay() == 0.0

        # After warmup starts, decay should increase but be capped by the max decay
        ema.optimization_step = torch.tensor(11)
        assert 0.0 < ema._get_current_decay() < 0.999

        # At a very large step, decay should be capped at the max value
        ema.optimization_step = torch.tensor(1_000_000)
        assert abs(ema._get_current_decay() - 0.999) < 1e-6

    @torch.no_grad()
    def test_update_logic(self, simple_model):
        """Tests the mathematical correctness of the EMA update rule."""
        original_val = simple_model.layer.weight.data.clone()
        ema = ExponentialMovingAverage(simple_model, decay=0.9, use_warmup=False)

        # First update
        new_val_1 = torch.ones_like(original_val)
        simple_model.layer.weight.data = new_val_1.clone()
        ema.update(simple_model)

        # Expected value: 0.9 * original_val + 0.1 * 1.0
        expected_ema_1 = 0.9 * original_val + (1 - 0.9) * new_val_1
        torch.testing.assert_close(ema.ema_model.layer.weight.data, expected_ema_1)
        assert ema.optimization_step == 1

        # Second update
        new_val_2 = torch.ones_like(original_val) * 2
        simple_model.layer.weight.data = new_val_2.clone()
        ema.update(simple_model)

        # Expected value: 0.9 * (previous_ema) + 0.1 * 2.0
        expected_ema_2 = 0.9 * expected_ema_1 + (1 - 0.9) * new_val_2
        torch.testing.assert_close(ema.ema_model.layer.weight.data, expected_ema_2)
        assert ema.optimization_step == 2

    def test_copy_to(self, simple_model):
        """Tests that the EMA parameters can be correctly copied to a model."""
        ema = ExponentialMovingAverage(simple_model)

        # Modify the EMA model's weights to a known value
        with torch.no_grad():
            ema.ema_model.layer.weight.data.fill_(123.0)

        # Copy to the original model
        ema.copy_to(simple_model)

        # Verify the original model now has the EMA's weights
        torch.testing.assert_close(
            simple_model.layer.weight.data, ema.ema_model.layer.weight.data
        )

    def test_average_parameters_context(self, simple_model):
        """Tests the context manager for safe evaluation."""
        original_weights = simple_model.layer.weight.data.clone()
        ema = ExponentialMovingAverage(simple_model)

        # Set EMA weights to a distinct value
        with torch.no_grad():
            ema.ema_model.layer.weight.data.fill_(555.0)

        # Use the context manager
        with ema.average_parameters_context(simple_model):
            # Inside the context, the model's weights should be the EMA weights
            assert torch.all(simple_model.layer.weight.data == 555.0)

        # Outside the context, the model's weights should be restored
        assert torch.all(simple_model.layer.weight.data == original_weights)

    def test_context_manager_restores_on_error(self, simple_model):
        """Ensures the context manager restores original weights even if an error occurs."""
        original_weights = simple_model.layer.weight.data.clone()
        ema = ExponentialMovingAverage(simple_model)
        with torch.no_grad():
            ema.ema_model.layer.weight.data.fill_(555.0)

        with pytest.raises(ValueError, match="Test error"):
            with ema.average_parameters_context(simple_model):
                # Weights are swapped to EMA weights here
                assert torch.all(simple_model.layer.weight.data == 555.0)
                # An error happens inside the block
                raise ValueError("Test error")

        # After the error, the original weights should still be restored
        assert torch.all(simple_model.layer.weight.data == original_weights)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_portability(self, simple_model):
        """Tests that the EMA module, including its buffer, moves to CUDA correctly."""
        ema = ExponentialMovingAverage(simple_model)
        ema.update(simple_model)  # step=1

        cuda_ema = ema.to("cuda")

        # Check that the EMA model and the optimization_step buffer are on the correct device
        assert cuda_ema.ema_model.layer.weight.device.type == "cuda"
        assert cuda_ema.optimization_step.device.type == "cuda"

        # Test updating on the new device
        cuda_model = simple_model.to("cuda")
        cuda_ema.update(cuda_model)
        assert cuda_ema.optimization_step == 2

    def test_state_dict_save_and_load(self, simple_model):
        """Tests that the EMA state can be correctly saved and loaded."""
        # Create and update an EMA instance
        ema1 = ExponentialMovingAverage(simple_model)
        with torch.no_grad():
            simple_model.layer.weight.data.fill_(1.0)
        ema1.update(simple_model)
        with torch.no_grad():
            simple_model.layer.weight.data.fill_(2.0)
        ema1.update(simple_model)

        # Save its state
        state_dict = ema1.state_dict()

        # Create a new EMA instance and load the state
        model2 = type(simple_model)()  # Create a fresh model
        ema2 = ExponentialMovingAverage(model2)
        ema2.load_state_dict(state_dict)

        # Verify that the state is restored
        assert ema2.optimization_step == ema1.optimization_step
        torch.testing.assert_close(
            ema2.ema_model.state_dict()["layer.weight"],
            ema1.ema_model.state_dict()["layer.weight"],
        )
