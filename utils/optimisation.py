"""Utilities for creating and managing optimizers and learning rate schedulers."""

from typing import Any, Dict, Optional, Union

from diffusers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType
from torch.optim import Optimizer


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Creates a learning rate scheduler instance from its name.

    This is a wrapper around the scheduler factory from the `diffusers` library,
    providing more robust argument validation and clear error messages.

    Args:
        name: The name of the scheduler to create (e.g., 'linear', 'cosine').
        optimizer: The optimizer that the scheduler will be applied to.
        num_warmup_steps: The number of steps for the warmup phase. Required by
            all schedulers except for 'constant'.
        num_training_steps: The total number of training steps. Required by
            schedulers that decay over the full training duration, such as
            'linear' and 'cosine'.
        **kwargs: Additional keyword arguments to be passed to the specific
            scheduler's constructor.

    Returns:
        An instantiated learning rate scheduler.

    Raises:
        ValueError: If a required argument for the specified scheduler
            (e.g., `num_warmup_steps`) is not provided.
        KeyError: If the provided scheduler name is not supported.
    """
    try:
        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    except (ValueError, KeyError) as e:
        raise KeyError(
            f"Scheduler '{name}' is not a valid SchedulerType. "
            f"Available types: {[st.value for st in SchedulerType]}"
        ) from e

    # Prepare arguments for the scheduler constructor.
    scheduler_kwargs: Dict[str, Any] = kwargs.copy()

    # Determine which arguments are required based on the scheduler type.
    # This approach is more scalable than a long if/elif chain.
    schedulers_requiring_warmup = [
        SchedulerType.LINEAR,
        SchedulerType.COSINE,
        SchedulerType.COSINE_WITH_RESTARTS,
        SchedulerType.POLYNOMIAL,
        SchedulerType.CONSTANT_WITH_WARMUP,
    ]
    schedulers_requiring_total_steps = [
        SchedulerType.LINEAR,
        SchedulerType.COSINE,
        SchedulerType.COSINE_WITH_RESTARTS,
        SchedulerType.POLYNOMIAL,
    ]

    if name in schedulers_requiring_warmup:
        if num_warmup_steps is None:
            raise ValueError(
                f"Scheduler '{name.value}' requires `num_warmup_steps`, but it "
                "was not provided."
            )
        scheduler_kwargs["num_warmup_steps"] = num_warmup_steps

    if name in schedulers_requiring_total_steps:
        if num_training_steps is None:
            raise ValueError(
                f"Scheduler '{name.value}' requires `num_training_steps`, but "
                "it was not provided."
            )
        scheduler_kwargs["num_training_steps"] = num_training_steps

    return schedule_func(optimizer, **scheduler_kwargs)
