from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class PushTConfig:
    """Configuration parameters for the Push-T environment."""

    # Simulation parameters
    sim_hz: int = 100
    control_hz: int = 10
    damping: float = 0.1
    gravity: Tuple[float, float] = (0.0, 0.0)

    # World and rendering parameters
    window_size: int = 512
    render_size: int = 96
    wall_thickness: float = 2.0
    wall_color: str = "LightGray"  # LightGray

    # Agent (Pusher) parameters
    agent_radius: float = 15.0
    agent_start_pos: Tuple[float, float] = (256.0, 400.0)
    agent_color: str = "RoyalBlue"  # RoyalBlue
    pd_gains: Tuple[float, float] = (100.0, 20.0)  # (k_p, k_v)

    # Block (T-Shape) parameters
    block_mass: float = 1.0
    block_friction: float = 0.7
    block_scale: float = 30.0
    block_color: str = "LightSlateGray"  # LightSlateGray

    # Goal parameters
    goal_color: str = "LightGreen"  # LightGreen
    goal_pose: np.ndarray = (256.0, 256.0, np.pi / 4)
    success_threshold: float = 0.95
