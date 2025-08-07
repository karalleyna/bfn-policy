"""A Gymnasium environment for the Push-T task.

This file defines the Push-T environment, which involves a kinematic pusher
(agent) attempting to push a T-shaped block to a target goal pose.

The implementation is modular, separating the physics simulation, rendering,
and the main environment logic into distinct classes. This approach promotes
code reuse, testability, and clarity.

Key Components:
  - PushTConfig: A dataclass to manage all simulation and environment constants.
  - PushTPhysicsSimulator: Handles all Pymunk physics calculations.
  - PushTRenderer: Manages all PyGame-based rendering.
  - PushTEnv: The main Gymnasium environment class that orchestrates the
    simulation and rendering components.
"""
import abc
import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.vec2d import Vec2d

from env_configs.pusht import PushTConfig
from renderers.pusht import PushTRenderer
from simulators.pusht import PushTPhysicsSimulator


class PushTEnv(gym.Env):
    """The main Push-T environment, orchestrating simulation and rendering.

    This class implements the Gymnasium API, combining the physics simulator
    and renderer to create a fully functional reinforcement learning environment.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[PushTConfig] = None,
        **kwargs,
    ):
        """Initializes the Push-T environment.

        Args:
          render_mode: The rendering mode ('human' or 'rgb_array').
          config: An optional configuration object. If None, a default
            PushTConfig is created.
          **kwargs: Keyword arguments to override fields in the config.
            For example, `render_size=128`. These are case-insensitive.
        """
        super().__init__()
        # Get the set of valid field names from the config dataclass.
        config_fields = {f.name for f in dataclasses.fields(PushTConfig)}

        # Filter and map kwargs to valid, uppercase config field names.
        override_kwargs = {}
        for key, value in kwargs.items():
            if key.lower() in config_fields:
                override_kwargs[key.lower()] = value
            else:
                # Raise an error for unexpected arguments, just like Python's
                # default behavior. This prevents silent errors.
                raise TypeError(
                    f"PushTEnv.__init__() got an unexpected keyword argument '{key}'"
                )

        # Start with a base config, either the one provided or a default one.
        base_config = config if config is not None else PushTConfig()

        # Create the final config by applying the overrides.
        self.config = dataclasses.replace(base_config, **override_kwargs)

        # The rest of the initialization proceeds with the final config.
        self.simulator = PushTPhysicsSimulator(self.config)
        self.renderer = PushTRenderer(self.config)

        self._goal_geom = self._precompute_goal_geometry()
        self.render_mode = render_mode

        # Define observation space: [agent_x, agent_y, block_x, block_y, block_angle]
        ws = self.config.window_size
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([ws, ws, ws, ws, 2 * np.pi], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )

        # Define action space: [target_agent_x, target_agent_y]
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([ws, ws], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def _precompute_goal_geometry(self) -> sg.MultiPolygon:
        """Pre-computes the Shapely geometry for the goal region.

        This avoids redundant calculations during the reward computation.

        Returns:
          A Shapely MultiPolygon representing the goal area.
        """
        goal_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        goal_body.position = self.config.goal_pose[:2]
        goal_body.angle = self.config.goal_pose[2]
        return self._pymunk_to_shapely(goal_body, self.simulator.block.shapes)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to a new random initial state."""
        super().reset(seed=seed)
        initial_state = self.observation_space.sample()
        self.simulator.set_state(initial_state)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step in the environment."""
        self.simulator.step(action)
        obs = self._get_obs()
        reward, done = self._calculate_reward()
        info = self._get_info()
        terminated = done
        truncated = False  # This environment does not have a time limit.
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment."""
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                'e.g. `gym.make("PushT", render_mode="human")`'
            )
            return None

        sim_state = self.simulator.get_state_dict()
        return self.renderer.render_frame(sim_state, self.render_mode)

    def close(self):
        """Closes the environment and its associated renderer."""
        self.renderer.close()

    def _get_obs(self) -> np.ndarray:
        """Constructs the observation array from the simulation state."""
        agent_pos = self.simulator.agent.position
        block_pos = self.simulator.block.position
        block_angle = self.simulator.block.angle % (2 * np.pi)
        return np.array(
            list(agent_pos) + list(block_pos) + [block_angle], dtype=np.float32
        )

    def _get_info(self) -> Dict[str, Any]:
        """Returns diagnostic information about the current step."""
        return {"n_contacts": self.simulator.n_contact_points}

    def _calculate_reward(self) -> Tuple[float, bool]:
        """Calculates reward based on Intersection over Union (IoU) of the block
        and goal geometries.

        Returns:
          A tuple containing the reward and a boolean indicating if the task is done.
        """
        block_geom = self._pymunk_to_shapely(
            self.simulator.block, self.simulator.block.shapes
        )
        intersection_area = self._goal_geom.intersection(block_geom).area
        union_area = self._goal_geom.union(block_geom).area

        iou = intersection_area / union_area if union_area > 0 else 0.0
        reward = np.clip(iou, 0.0, 1.0)
        done = iou > self.config.success_threshold
        return reward, done

    @staticmethod
    def _pymunk_to_shapely(
        body: pymunk.Body, shapes: List[pymunk.Shape]
    ) -> sg.MultiPolygon:
        """Converts a Pymunk body's shapes into a single Shapely MultiPolygon.

        Args:
          body: The Pymunk body.
          shapes: The list of Pymunk shapes attached to the body.

        Returns:
          A Shapely MultiPolygon representing the combined geometry.
        """
        geoms = []
        for shape in shapes:
            if isinstance(shape, pymunk.Poly):
                # Transform vertices from local body coordinates to world coordinates.
                world_coords = [body.local_to_world(v) for v in shape.get_vertices()]
                geoms.append(sg.Polygon(world_coords))
        return sg.MultiPolygon(geoms)
