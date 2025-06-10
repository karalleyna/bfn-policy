import copy
from contextlib import contextmanager
from typing import Optional

import torch
from torch import nn


class ExponentialMovingAverage(nn.Module):
    """
    Maintains an Exponential Moving Average (EMA) of a model's parameters.

    This class creates a "shadow" copy of the model's parameters and updates
    them with a smoothed average of the training model's parameters at each
    step. This often leads to better performance and more stable models during
    evaluation.

    The EMA decay rate is calculated with a warmup schedule, as described in
    the original implementation notes, to prevent large parameter swings early
    in training.

    Usage:
        # 1. Initialize EMA after creating the model
        model = MyModel()
        ema = ExponentialMovingAverage(model, decay=0.999)

        # 2. In your training loop, after each optimizer step:
        optimizer.step()
        ema.update(model)

        # 3. For evaluation, use the context manager to temporarily
        #    load the averaged parameters into your model.
        with ema.average_parameters_context():
            # All code in this block uses the EMA model for evaluation
            val_loss = evaluate(model)

    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        use_warmup: bool = True,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
    ):
        """
        Initializes the EMA tracker.

        Args:
            model: The model whose parameters will be tracked.
            decay: The base decay rate for the EMA. If `use_warmup` is False,
                   this value is used directly.
            update_after_step: Start updating the EMA after this many steps.
            use_warmup: If True, uses a warmup schedule for the decay rate.
            inv_gamma: The inverse gamma factor for the warmup schedule.
            power: The power factor for the warmup schedule.
        """
        super().__init__()
        # Create a deep copy of the model for the EMA weights, detached from the graph
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

        self.decay = decay
        self.update_after_step = update_after_step

        # Warmup schedule parameters
        self.use_warmup = use_warmup
        self.inv_gamma = inv_gamma
        self.power = power

        # Use register_buffer for stateful attributes that are not model parameters.
        # This ensures they are saved in the state_dict and moved to the correct device.
        self.register_buffer("optimization_step", torch.tensor(0, dtype=torch.long))

    def _get_current_decay(self) -> float:
        """Calculates the decay rate for the current optimization step."""
        if not self.use_warmup:
            return self.decay

        step = self.optimization_step.item()
        if step < self.update_after_step:
            return 0.0

        # Warmup schedule from original implementation
        warmup_step = step - self.update_after_step
        value = 1.0 - (1.0 + warmup_step / self.inv_gamma) ** -self.power
        return max(0.0, min(self.decay, value))

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Updates the EMA parameters from the live model.

        This method should be called after each training step (e.g., after
        `optimizer.step()`).

        Args:
            model: The current training model with updated weights.
        """
        current_decay = self._get_current_decay()

        # Update the EMA model's state_dict
        ema_state_dict = self.ema_model.state_dict()
        model_state_dict = model.state_dict()

        for key in model_state_dict:
            # The update rule is:
            # ema_param = decay * ema_param + (1 - decay) * model_param
            ema_param = ema_state_dict[key].to(torch.float32)
            model_param = model_state_dict[key].to(torch.float32)

            ema_param.mul_(current_decay).add_(model_param, alpha=1.0 - current_decay)

        self.optimization_step += 1

    def copy_to(self, model: nn.Module):
        """
        Copies the EMA parameters to the provided model.

        This is useful for evaluation where you want to use the averaged
        weights directly.

        Args:
            model: The model to copy the EMA weights into.
        """
        model.load_state_dict(self.ema_model.state_dict())

    @contextmanager
    def average_parameters_context(self, model: nn.Module):
        """
        A context manager to temporarily load the EMA parameters into a model
        for inference or evaluation.

        This is the recommended way to use the EMA model for evaluation, as it
        safely restores the original model weights afterwards.

        Args:
            model: The model to load the EMA weights into.
        """
        # 1. Save the original parameters
        original_state_dict = copy.deepcopy(model.state_dict())

        # 2. Copy EMA parameters to the model
        self.copy_to(model)

        try:
            # 3. Yield control to the user's code block
            yield
        finally:
            # 4. Restore the original parameters, even if an error occurred
            model.load_state_dict(original_state_dict)
