import abc
import dataclasses
from typing import Any, Dict, Final, List, Optional, Tuple

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


class PushTPhysicsSimulator:
    """Handles all PyMunk physics simulation logic for the Push-T task.

    This class is self-contained and only concerns itself with the physics state,
    independent of rendering or rewards.
    """

    def __init__(self, config: PushTConfig):
        """Initializes the physics simulation.

        Args:
            config: The environment configuration object.
        """
        self.config: Final[PushTConfig] = config
        self.space: Final[pymunk.Space] = self._create_space()
        self.agent: Final[pymunk.Body] = self._create_agent()
        self.block: Final[pymunk.Body] = self._create_block()
        self.n_contact_points: int = 0
        self._setup_collision_handler()

    def _create_space(self) -> pymunk.Space:
        """Creates and configures the Pymunk simulation space."""
        space = pymunk.Space()
        space.gravity = self.config.gravity
        space.damping = self.config.damping

        # Add static walls around the perimeter.
        s = self.config.window_size
        t = self.config.wall_thickness
        static_lines = [
            pymunk.Segment(space.static_body, (t, s - t), (t, t), t),
            pymunk.Segment(space.static_body, (t, t), (s - t, t), t),
            pymunk.Segment(space.static_body, (s - t, t), (s - t, s - t), t),
            pymunk.Segment(space.static_body, (t, s - t), (s - t, s - t), t),
        ]
        for line in static_lines:
            line.color = pygame.Color(self.config.wall_color)
        space.add(*static_lines)
        return space

    def _create_agent(self) -> pymunk.Body:
        """Creates the agent (pusher) body and shape."""
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = self.config.agent_start_pos
        shape = pymunk.Circle(body, self.config.agent_radius)
        shape.color = pygame.Color(self.config.agent_color)
        self.space.add(body, shape)
        return body

    def _create_block(self) -> pymunk.Body:
        """Creates the T-shaped block from two rectangles."""
        mass = self.config.block_mass
        scale = self.config.block_scale

        # Define the two parts of the 'T'
        rect1_verts = [
            (-2 * scale, scale),
            (2 * scale, scale),
            (2 * scale, 0),
            (-2 * scale, 0),
        ]
        rect2_verts = [
            (-scale / 2, scale),
            (-scale / 2, 4 * scale),
            (scale / 2, 4 * scale),
            (scale / 2, scale),
        ]

        # Pymunk can correctly calculate the combined moment of inertia
        # and center of gravity for a compound body.
        moment = pymunk.moment_for_poly(
            mass / 2, vertices=rect1_verts
        ) + pymunk.moment_for_poly(mass / 2, vertices=rect2_verts)

        body = pymunk.Body(mass, moment)
        shape1 = pymunk.Poly(body, rect1_verts)
        shape2 = pymunk.Poly(body, rect2_verts)

        # Set shared properties
        for shape in [shape1, shape2]:
            shape.friction = self.config.block_friction
            shape.color = pygame.Color(self.config.block_color)

        body.position = self.config.window_size / 2, self.config.window_size * 0.75
        self.space.add(body, shape1, shape2)
        return body

    def _collision_callback(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: Dict
    ) -> bool:
        """Callback to count contact points between bodies."""
        self.n_contact_points += len(arbiter.contact_point_set.points)
        return True

    def _setup_collision_handler(self) -> None:
        """Sets up the collision handler to monitor agent-block interaction."""
        handler = self.space.add_default_collision_handler()
        handler.post_solve = self._collision_callback

    def set_state(self, state: np.ndarray) -> None:
        """Sets the simulation state from a NumPy array.

        Args:
            state: A 5D array [agent_x, agent_y, block_x, block_y, block_angle].
        """
        self.agent.position = state[0], state[1]

        # For compound bodies in PyMunk, it's crucial to set the position *before*
        # the angle to ensure the body rotates around its correct center of gravity.
        self.block.position = state[2], state[3]
        self.block.angle = state[4]

        # Reset velocities to ensure a clean state
        self.agent.velocity = (0, 0)
        self.block.velocity = (0, 0)
        self.block.angular_velocity = 0

        # Step the space once to update the state.
        self.space.step(1.0 / self.config.sim_hz)

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns the current state of the simulation bodies for rendering."""
        return {"agent": self.agent, "block": self.block, "space": self.space}

    def step(self, action: np.ndarray) -> None:
        """Steps the simulation forward using PD control.

        Args:
            action: The target position for the agent [x, y].
            n_sub_steps: The number of physics steps to perform.
        """
        dt = 1.0 / self.config.sim_hz
        k_p, k_v = self.config.pd_gains
        self.n_contact_points = 0

        n_sub_steps = self.config.sim_hz // self.config.control_hz
        for _ in range(n_sub_steps):
            # Simple PD controller to move the kinematic agent smoothly.
            position_error = action - self.agent.position
            velocity_error = Vec2d(0, 0) - self.agent.velocity
            acceleration = (k_p * position_error) + (k_v * velocity_error)
            self.agent.velocity += acceleration * dt
            self.space.step(dt)
