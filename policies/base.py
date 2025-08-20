import abc
from typing import Any, Dict

import torch
import torch.nn as nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for all policies.

    This class defines the essential interface for a policy, ensuring that all
    subclasses implement the necessary methods for action prediction, loss
    computation, and state management. This modular design promotes consistency
    and reusability across different policy implementations.
    """

    def __init__(self):
        """
        Initializes the BasePolicy.
        """
        super().__init__()

    @abc.abstractmethod
    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Predicts an action based on the provided observation.

        This method encapsulates the core logic of the policy, mapping observations
        to actions. It is designed to be used during inference and evaluation.

        Args:
            obs_dict: A dictionary of observation tensors, where keys are modality
                      names and values are tensors of shape (B, ...), where B is
                      the batch size.

        Returns:
            A dictionary containing the predicted action tensors. The keys should
            be descriptive of the action components (e.g., 'action', 'action_logits').
        """
        raise NotImplementedError("Subclasses must implement predict_action.")

    @abc.abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the loss for a given batch of data.

        This method is central to the training process. It takes a batch of
        experience, which includes observations and ground-truth actions, and
        computes a scalar loss value to be used for backpropagation.

        Args:
            batch: A dictionary of tensors, typically containing 'obs', 'action',
                   and other relevant data for loss computation.

        Returns:
            A scalar tensor representing the computed loss.
        """
        raise NotImplementedError("Subclasses must implement compute_loss.")

    def reset(self):
        """
        Resets the internal state of the policy.

        This is particularly useful for stateful policies, such as those using
        recurrent neural networks (RNNs), which need to be reset at the beginning
        of each episode. By default, this method does nothing, but it can be
        overridden by subclasses that require state management.
        """
        pass

    def forward(
        self, obs_dict: Dict[str, torch.Tensor], *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Defines the forward pass of the policy.

        By default, this method calls `predict_action` to maintain a consistent
        interface. This allows the policy to be used as a standard `nn.Module`
        in PyTorch, for example, within a larger network or for tracing with
        TorchScript.

        Args:
            obs_dict: A dictionary of observation tensors.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing the predicted action tensors.
        """
        return self.predict_action(obs_dict)
