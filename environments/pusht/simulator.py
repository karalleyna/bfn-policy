"""
Physics simulator for the Push-T environment using Pymunk.

This module handles the creation of the physics space, objects (agent, block),
and stepping the simulation forward.
"""
import numpy as np
import pymunk
from pymunk.vec2d import Vec2d

from .config import PushTConfig


class PushTPhysicsSimulator:
    """Manages the Pymunk physics simulation for the Push-T task."""

    def __init__(self, config: PushTConfig):
        """Initializes the physics world."""
        self.config = config
        self._setup_space()

    def _setup_space(self):
        """Creates the Pymunk space and all static and dynamic bodies."""
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = self.config.damping

        # Add walls
        walls = [
            self._add_segment((5, 506), (5, 5)),
            self._add_segment((5, 5), (506, 5)),
            self._add_segment((506, 5), (506, 506)),
            self._add_segment((5, 506), (506, 506)),
        ]
        self.space.add(*walls)

        # Add dynamic bodies
        self.agent = self._add_agent()
        self.block = self._add_tee()

    def step(self, action: np.ndarray):
        """
        Steps the simulation forward for one control step.

        Args:
            action: The target position for the agent's PD controller.
        """
        dt = 1.0 / self.config.sim_hz
        n_steps = self.config.sim_hz // self.config.control_hz

        for _ in range(n_steps):
            # PD controller for the agent
            acceleration = self.config.pd_k_p * (
                action - self.agent.position
            ) + self.config.pd_k_v * (Vec2d(0, 0) - self.agent.velocity)
            self.agent.velocity += acceleration * dt
            self.space.step(dt)

    def set_state(self, state: np.ndarray):
        """
        Sets the pose of the agent and block.

        Args:
            state: A 5D numpy array [agent_x, agent_y, block_x, block_y, block_angle].
        """
        self.agent.position = state[:2]
        self.agent.velocity = (0, 0)
        self.block.position = state[2:4]
        self.block.angle = state[4]
        self.block.velocity = (0, 0)
        self.block.angular_velocity = 0
        # Step once to let the changes settle
        self.space.step(1.0 / self.config.sim_hz)

    def get_state_dict(self) -> dict:
        """Returns a dictionary with the current state of simulation objects."""
        return {
            "agent_body": self.agent,
            "block_body": self.block,
            "static_bodies": self.space.static_body,
        }

    def _add_segment(self, a: tuple, b: tuple) -> pymunk.Segment:
        """Helper to create a wall segment."""
        shape = pymunk.Segment(self.space.static_body, a, b, self.config.wall_thickness)
        shape.color = self.config.wall_color
        return shape

    def _add_agent(self) -> pymunk.Body:
        """Helper to create the agent body and shape."""
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        shape = pymunk.Circle(body, self.config.agent_radius)
        shape.color = self.config.agent_color
        self.space.add(body, shape)
        return body

    def _add_tee(self) -> pymunk.Body:
        """Helper to create the T-shaped block."""
        mass = self.config.block_mass
        scale = self.config.block_scale

        # Define the two rectangular parts of the T-shape
        bar1_dims = (4 * scale, scale)
        bar2_dims = (scale, 4 * scale)

        # Create vertices for the two parts
        vs1 = [
            (-bar1_dims[0] / 2, bar1_dims[1] / 2),
            (bar1_dims[0] / 2, bar1_dims[1] / 2),
            (bar1_dims[0] / 2, -bar1_dims[1] / 2),
            (-bar1_dims[0] / 2, -bar1_dims[1] / 2),
        ]

        vs2 = [
            (-bar2_dims[0] / 2, bar2_dims[1] / 2),
            (bar2_dims[0] / 2, bar2_dims[1] / 2),
            (bar2_dims[0] / 2, -bar2_dims[1] / 2),
            (-bar2_dims[0] / 2, -bar2_dims[1] / 2),
        ]

        # Adjust position of the second bar to form a T
        vs2 = [(x, y + scale * 1.5) for x, y in vs2]

        # Create the body with combined moment of inertia
        moment = pymunk.moment_for_poly(
            mass / 2, vertices=vs1
        ) + pymunk.moment_for_poly(mass / 2, vertices=vs2)
        body = pymunk.Body(mass, moment)

        # Create shapes and attach to the body
        shape1 = pymunk.Poly(body, vs1)
        shape2 = pymunk.Poly(body, vs2)
        shape1.color = self.config.block_color
        shape2.color = self.config.block_color

        self.space.add(body, shape1, shape2)
        return body
