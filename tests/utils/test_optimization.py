"""Unit tests for optimization utilities."""

from unittest.mock import patch

import pytest
import torch
from diffusers.optimization import SchedulerType
from torch import nn

# Assuming the utility is in utils/optimization.py
from utils.optimisation import get_scheduler


@pytest.fixture
def optimizer():
    """A simple optimizer fixture for testing."""
    model = nn.Linear(10, 10)
    return torch.optim.AdamW(model.parameters(), lr=1e-4)


def test_invalid_scheduler_name(optimizer):
    """Tests that an invalid scheduler name raises a KeyError."""
    with pytest.raises(KeyError) as excinfo:
        get_scheduler("invalid_scheduler", optimizer)
    assert "Scheduler 'invalid_scheduler' is not a valid SchedulerType" in str(
        excinfo.value
    )


@pytest.mark.parametrize(
    "scheduler_type",
    [
        SchedulerType.LINEAR,
        SchedulerType.COSINE,
        SchedulerType.POLYNOMIAL,
        SchedulerType.COSINE_WITH_RESTARTS,
    ],
)
def test_schedulers_requiring_all_args(optimizer, scheduler_type):
    """Tests schedulers that require both warmup and total training steps."""
    # Test successful creation
    scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=1000,
    )
    assert scheduler is not None

    # Test failure without num_warmup_steps
    with pytest.raises(ValueError) as excinfo:
        get_scheduler(scheduler_type, optimizer=optimizer, num_training_steps=1000)
    assert f"Scheduler '{scheduler_type.value}' requires `num_warmup_steps`" in str(
        excinfo.value
    )

    # Test failure without num_training_steps
    with pytest.raises(ValueError) as excinfo:
        get_scheduler(scheduler_type, optimizer=optimizer, num_warmup_steps=100)
    assert f"Scheduler '{scheduler_type.value}' requires `num_training_steps`" in str(
        excinfo.value
    )


def test_constant_with_warmup_scheduler(optimizer):
    """Tests the 'constant_with_warmup' scheduler."""
    scheduler_type = SchedulerType.CONSTANT_WITH_WARMUP

    # Test successful creation
    scheduler = get_scheduler(scheduler_type, optimizer=optimizer, num_warmup_steps=100)
    assert scheduler is not None

    # Test failure without num_warmup_steps
    with pytest.raises(ValueError) as excinfo:
        get_scheduler(scheduler_type, optimizer=optimizer)
    assert f"Scheduler '{scheduler_type.value}' requires `num_warmup_steps`" in str(
        excinfo.value
    )


def test_constant_scheduler(optimizer):
    """Tests the 'constant' scheduler, which requires no extra arguments."""
    scheduler = get_scheduler(SchedulerType.CONSTANT, optimizer=optimizer)
    assert scheduler is not None


@patch("utils.optimisation.TYPE_TO_SCHEDULER_FUNCTION")
def test_kwargs_passthrough(mock_scheduler_map, optimizer):
    """Tests that additional kwargs are passed to the underlying scheduler factory."""
    scheduler_type = SchedulerType.COSINE_WITH_RESTARTS
    mock_factory_func = mock_scheduler_map.__getitem__.return_value

    # Call our wrapper function
    get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=1000,
        num_cycles=5,  # This is the kwarg we want to test
    )

    # Assert that our wrapper called the underlying factory from diffusers
    # with all the correct arguments, including the extra kwarg.
    mock_factory_func.assert_called_once_with(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=1000,
        num_cycles=5,
    )
