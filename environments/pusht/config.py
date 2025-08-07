"""
Configuration for the Push-T Environment.

This module uses a dataclass to hold all physical and visual parameters,
allowing for easy and type-safe configuration of the environment.
"""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from pygame.color import Color


@dataclass
class PushTConfig:
    """Configuration parameters for the Push-T environment."""

    # --- Simulation Parameters ---
    sim_hz: int = 100
    control_hz: int = 10
    damping: float = 0.1
    pd_k_p: float = 100.0
    pd_k_v: float = 20.0

    # --- Arena and Object Parameters ---
    window_size: int = 512
    agent_radius: float = 15.0
    block_scale: float = 30.0
    block_mass: float = 1.0
    block_friction: float = 0.7  # Added friction parameter
    wall_thickness: float = 2.0
    goal_pose: np.ndarray = field(
        default_factory=lambda: np.array([256.0, 256.0, np.pi / 4.0])
    )
    success_threshold: float = 0.95

    # --- Rendering Parameters ---
    render_size: int = 96
    render_action: bool = True

    # --- Colors ---
    agent_color: Color = field(default_factory=lambda: Color("RoyalBlue"))
    block_color: Color = field(default_factory=lambda: Color("LightSlateGray"))
    goal_color: Color = field(default_factory=lambda: Color("LightGreen"))
    wall_color: Color = field(default_factory=lambda: Color("LightGray"))
    action_color: Tuple[int, int, int] = (255, 0, 0)  # Red for action marker
