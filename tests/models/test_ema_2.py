"""Unit tests for the ExponentialMovingAverage utility."""

import pytest
import torch
from torch import nn

# Assuming the utility is in models/ema.py
from models.ema import ExponentialMovingAverage


@pytest.fixture
def simple_model():
    """Provides a basic nn.Module for testing."""
    return nn.Linear(10, 10)


def test_initialization(simple_model):
    """Tests that the EMA is initialized with a correct copy of the model."""
    ema = ExponentialMovingAverage(simple_model)

    assert isinstance(ema.ema_model, nn.Module)
    for ema_param, model_param in zip(
        ema.ema_model.parameters(), simple_model.parameters()
    ):
        assert torch.equal(ema_param, model_param)


def test_update_no_warmup(simple_model):
    """Tests a single update step of the EMA parameters WITHOUT warmup."""
    with torch.no_grad():
        for param in simple_model.parameters():
            param.fill_(1.0)

    # FIX: Explicitly disable warmup to test the constant decay path.
    ema = ExponentialMovingAverage(simple_model, decay=0.9, use_warmup=False)
    ema.optimization_step.fill_(
        1
    )  # Step once to ensure update_after_step (default 0) is passed

    with torch.no_grad():
        for param in simple_model.parameters():
            param.fill_(0.0)

    ema.update(simple_model)

    # Expected value = 0.9 * 1.0 + (1 - 0.9) * 0.0 = 0.9
    for ema_param in ema.ema_model.parameters():
        assert torch.allclose(ema_param, torch.full_like(ema_param, 0.9))
    assert ema.optimization_step == 2


def test_update_with_warmup(simple_model):
    """Tests a single update step with warmup enabled to ensure correct decay calculation."""
    with torch.no_grad():
        for param in simple_model.parameters():
            param.fill_(1.0)

    # Initialize with warmup enabled (default)
    ema = ExponentialMovingAverage(simple_model, decay=0.9)
    ema.optimization_step.fill_(1)  # Step once

    with torch.no_grad():
        for param in simple_model.parameters():
            param.fill_(0.0)

    # Calculate expected decay based on the warmup formula
    expected_decay = ema._get_current_decay()

    ema.update(simple_model)

    expected_value = expected_decay * 1.0 + (1 - expected_decay) * 0.0
    for ema_param in ema.ema_model.parameters():
        assert torch.allclose(ema_param, torch.full_like(ema_param, expected_value))


def test_warmup_decay_schedule(simple_model):
    """Tests the behavior of the warmup decay schedule."""
    ema = ExponentialMovingAverage(
        simple_model, decay=0.999, update_after_step=10, use_warmup=True
    )

    assert ema._get_current_decay() == 0.0

    ema.optimization_step.fill_(15)
    decay1 = ema._get_current_decay()
    assert 0 < decay1 < 0.999

    ema.optimization_step.fill_(20)
    decay2 = ema._get_current_decay()
    assert decay1 < decay2 < 0.999

    ema.optimization_step.fill_(100000)  # Increased steps for precision
    # FIX: Use pytest.approx for floating point comparison of asymptotic value
    assert ema._get_current_decay() == pytest.approx(0.999, abs=1e-4)


def test_no_warmup_with_update_step(simple_model):
    """Tests that decay is constant when warmup is disabled but update_after_step is set."""
    ema = ExponentialMovingAverage(
        simple_model, decay=0.9, use_warmup=False, update_after_step=5
    )
    # At step 0, it should be 0.0
    assert ema._get_current_decay() == 0.0
    # After update_after_step, it should be the constant decay value
    ema.optimization_step.fill_(10)
    assert ema._get_current_decay() == 0.9


def test_copy_to(simple_model):
    """Tests copying EMA parameters to a model."""
    ema = ExponentialMovingAverage(simple_model)
    with torch.no_grad():
        for param in ema.ema_model.parameters():
            param.fill_(123.0)
    ema.copy_to(simple_model)
    for model_param in simple_model.parameters():
        assert torch.all(torch.eq(model_param, 123.0))


def test_average_parameters_context(simple_model):
    """Tests the context manager for evaluation."""
    original_params = [p.clone() for p in simple_model.parameters()]
    ema = ExponentialMovingAverage(simple_model)
    with torch.no_grad():
        for param in ema.ema_model.parameters():
            param.fill_(456.0)
    with ema.average_parameters_context(simple_model):
        for model_param in simple_model.parameters():
            assert torch.all(torch.eq(model_param, 456.0))
    for model_param, original_param in zip(simple_model.parameters(), original_params):
        assert torch.equal(model_param, original_param)


def test_average_parameters_context_with_error(simple_model):
    """Ensures original parameters are restored even if an error occurs."""
    original_params = [p.clone() for p in simple_model.parameters()]
    ema = ExponentialMovingAverage(simple_model)
    with torch.no_grad():
        for param in ema.ema_model.parameters():
            param.fill_(789.0)
    try:
        with ema.average_parameters_context(simple_model):
            raise ValueError("Simulating an error during evaluation")
    except ValueError:
        pass
    for model_param, original_param in zip(simple_model.parameters(), original_params):
        assert torch.equal(model_param, original_param)
