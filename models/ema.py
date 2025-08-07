"""
Implements a robust Exponential Moving Average (EMA) for PyTorch models.
"""

import copy
from contextlib import contextmanager
from typing import Optional

import torch
from torch import nn


class ExponentialMovingAverage(nn.Module):
    """
    Maintains an Exponential Moving Average (EMA) of a model's parameters.
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
        """Initializes the EMA tracker."""
        super().__init__()
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

        self.decay = decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.inv_gamma = inv_gamma
        self.power = power

        self.register_buffer("optimization_step", torch.tensor(0, dtype=torch.long))

    def _get_current_decay(self) -> float:
        """Calculates the decay rate for the current optimization step."""
        step = self.optimization_step.item()

        # This check must come first, regardless of warmup settings.
        if step < self.update_after_step:
            return 0.0

        if not self.use_warmup:
            return self.decay

        # Warmup schedule from original implementation
        warmup_step = step - self.update_after_step
        value = 1.0 - (1.0 + warmup_step / self.inv_gamma) ** -self.power
        return max(0.0, min(self.decay, value))

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Updates the EMA parameters from the live model."""
        current_decay = self._get_current_decay()

        ema_state_dict = self.ema_model.state_dict()
        model_state_dict = model.state_dict()

        for key in model_state_dict:
            ema_param = ema_state_dict[key].to(torch.float32)
            model_param = model_state_dict[key].to(torch.float32)
            ema_param.mul_(current_decay).add_(model_param, alpha=1.0 - current_decay)

        self.optimization_step += 1

    def copy_to(self, model: nn.Module):
        """Copies the EMA parameters to the provided model."""
        model.load_state_dict(self.ema_model.state_dict())

    @contextmanager
    def average_parameters_context(self, model: nn.Module):
        """A context manager to temporarily load the EMA parameters."""
        original_state_dict = copy.deepcopy(model.state_dict())
        self.copy_to(model)
        try:
            yield
        finally:
            model.load_state_dict(original_state_dict)
