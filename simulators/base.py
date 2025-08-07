# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the abstract base class for a physics/state simulator."""
from __future__ import annotations

import abc
from typing import Any, Dict, Generic, TypeVar

import numpy as np

from .base_config import BaseConfig

# A TypeVar for generic dataclass configurations.
ConfigType = TypeVar("ConfigType", bound=BaseConfig)


class BaseSimulator(abc.ABC, Generic[ConfigType]):
    """Abstract base class for a physics simulator.

    This class defines the essential interface for a simulation engine that
    manages the environment's state, independent of rendering or rewards.
    """

    def __init__(self, config: ConfigType):
        """Initializes the simulator.

        Args:
            config: The environment configuration object.
        """
        self.config = config

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> None:
        """Advances the simulation by one time step.

        Args:
            action: The action to apply to the agent(s).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        """Sets the entire simulation to a specific state.

        Args:
            state: A numerical representation of the desired environment state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """Returns the raw simulation state for observation or rendering.

        Returns:
            A dictionary containing the core simulation objects (e.g., Pymunk bodies).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_obs(self) -> np.ndarray:
        """Returns the current observation as a NumPy array.

        Returns:
            The observation array.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns a dictionary of diagnostic information.

        Returns:
            A dictionary with auxiliary information about the simulation state.
        """
        raise NotImplementedError
