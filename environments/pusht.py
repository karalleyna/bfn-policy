import abc
import collections
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.vec2d import Vec2d

# =========================== 1. ABSTRACT BASE ENVIRONMENT ===========================


class BaseEnv(gym.Env, abc.ABC):
    """
    Abstract base class for all environments in this project.
    It enforces the standard Gymnasium API.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}
    reward_range = (0.0, 1.0)


# ======================== 2. SIMULATION LOGIC COMPONENT =======================


class PushTPhysicsSimulator:
    """
    Handles all PyMunk physics simulation logic for the Push-T task.
    This class is self-contained and knows nothing about rendering or rewards.
    """

    def __init__(
        self, sim_hz: int = 100, damping: float = 0.1, block_friction: float = 0.7
    ):
        self.sim_hz = sim_hz
        self.damping = damping
        self.block_friction = block_friction
        self.space = self._create_space()
        self.agent = self._create_agent()
        self.block = self._create_block()
        self.n_contact_points = 0
        self._setup_collision_handler()

    def _create_space(self) -> pymunk.Space:
        """Creates and configures the Pymunk simulation space."""
        space = pymunk.Space()
        space.gravity = 0, 0
        space.damping = self.damping

        # Add static walls
        walls = [
            pymunk.Segment(space.static_body, (5, 506), (5, 5), 2),
            pymunk.Segment(space.static_body, (5, 5), (506, 5), 2),
            pymunk.Segment(space.static_body, (506, 5), (506, 506), 2),
            pymunk.Segment(space.static_body, (5, 506), (506, 506), 2),
        ]
        for wall in walls:
            wall.color = pygame.Color("LightGray")
        space.add(*walls)
        return space

    def _create_agent(self) -> pymunk.Body:
        """Creates the agent (pusher) body and shape."""
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (256, 400)
        shape = pymunk.Circle(body, 15)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def _create_block(self) -> pymunk.Body:
        """Creates the T-shaped block body and shapes."""
        mass = 1.0
        scale = 30
        length = 4

        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass / 2, vertices=vertices1)

        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass / 2, vertices=vertices2)

        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)

        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2

        shape1.friction = self.block_friction
        shape2.friction = self.block_friction

        shape1.color = pygame.Color("LightSlateGray")
        shape2.color = pygame.Color("LightSlateGray")
        body.position = (256, 300)
        self.space.add(body, shape1, shape2)
        return body

    def _handle_collision(self, arbiter, space, data):
        """Callback to count contact points during simulation."""
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _setup_collision_handler(self):
        """Sets up the collision handler."""
        handler = self.space.add_collision_handler(0, 0)
        handler.post_solve = self._handle_collision

    def set_state(self, state: np.ndarray):
        """
        Sets the state of the simulation.

        Args:
            state: A 5D array [agent_x, agent_y, block_x, block_y, block_angle].
        """
        self.agent.position = state[:2].tolist()

        # --- FIX IS HERE: Set position BEFORE angle for compound bodies ---
        # This ensures the body is placed correctly before being rotated around its
        # potentially non-zero center of gravity.
        self.block.position = state[2:4].tolist()
        self.block.angle = state[4]

        self.agent.velocity = (0, 0)
        self.block.velocity = (0, 0)
        self.block.angular_velocity = 0
        self.space.step(1.0 / self.sim_hz)

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the simulation bodies."""
        return {"agent": self.agent, "block": self.block, "space": self.space}

    def step(self, action: np.ndarray, n_sub_steps: int):
        """Steps the simulation forward using PD control."""
        dt = 1.0 / self.sim_hz
        k_p, k_v = 100, 20
        self.n_contact_points = 0

        for _ in range(n_sub_steps):
            acceleration = k_p * (action - self.agent.position) + k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * dt
            self.space.step(dt)


# =========================== 3. RENDERING LOGIC COMPONENT ==========================


class PushTRenderer:
    """Handles all PyGame rendering for the Push-T task."""

    def __init__(self, render_size: int, window_size: int = 512):
        self.render_size = render_size
        self.window_size = window_size
        self.window = None
        self.clock = None

    def _initialize_pygame(self):
        """Initializes PyGame and the display window if not already done."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render_frame(
        self, sim_state: Dict[str, Any], goal_pose: np.ndarray, mode: str
    ) -> np.ndarray:
        """Renders a single frame of the environment."""
        self._initialize_pygame()
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        draw_options = pymunk.pygame_util.DrawOptions(canvas)
        self._draw_goal(canvas, sim_state["block"], goal_pose)
        sim_state["space"].debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(30)

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        return cv2.resize(img, (self.render_size, self.render_size))

    def _draw_goal(
        self, surface: pygame.Surface, block_body: pymunk.Body, goal_pose: np.ndarray
    ):
        """Helper to draw the green goal area."""
        goal_body_static = pymunk.Body(body_type=pymunk.Body.STATIC)
        goal_body_static.position = goal_pose[:2].tolist()
        goal_body_static.angle = goal_pose[2]

        for shape in block_body.shapes:
            verts = [goal_body_static.local_to_world(v) for v in shape.get_vertices()]
            pygame.draw.polygon(surface, pygame.Color("LightGreen"), verts)

    def close(self):
        """Closes the PyGame window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


# ========================= 4. THE MAIN ENVIRONMENT CLASS =========================


class PushTEnv(BaseEnv):
    """
    The main Push-T environment class, orchestrating the simulator and renderer.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        render_size: int = 96,
        success_threshold: float = 0.95,
        damping: float = 0.1,
        block_friction: float = 0.7,
    ):
        super().__init__()

        self.render_size = render_size
        self.success_threshold = success_threshold
        self.control_hz = self.metadata["video.frames_per_second"]
        self.goal_pose = np.array([256, 256, np.pi / 4])

        self.simulator = PushTPhysicsSimulator(
            damping=damping, block_friction=block_friction
        )
        self.renderer = PushTRenderer(render_size)

        ws = 512
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([ws, ws, ws, ws, np.pi * 2], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([ws, ws], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        state = self.observation_space.sample()
        self.simulator.set_state(state)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        n_sub_steps = self.simulator.sim_hz // self.control_hz
        self.simulator.step(action, n_sub_steps)
        reward, done = self._calculate_reward()
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info

    def render(self) -> np.ndarray:
        sim_state = self.simulator.get_state()
        return self.renderer.render_frame(sim_state, self.goal_pose, self.render_mode)

    def close(self):
        self.renderer.close()

    def _get_obs(self) -> np.ndarray:
        agent = self.simulator.agent
        block = self.simulator.block
        return np.array(
            list(agent.position) + list(block.position) + [block.angle % (2 * np.pi)],
            dtype=np.float32,
        )

    def _get_info(self) -> Dict[str, Any]:
        return {"n_contacts": self.simulator.n_contact_points}

    def _calculate_reward(self) -> Tuple[float, bool]:
        """Calculates reward based on block coverage of the goal area."""
        block_body = self.simulator.block

        goal_body_static = pymunk.Body(body_type=pymunk.Body.STATIC)
        goal_body_static.position = self.goal_pose[:2].tolist()
        goal_body_static.angle = self.goal_pose[2]

        goal_shapes = [
            pymunk.Poly(goal_body_static, poly.get_vertices())
            for poly in block_body.shapes
        ]

        goal_geom = self._pymunk_to_shapely(goal_body_static, goal_shapes)
        block_geom = self._pymunk_to_shapely(block_body, block_body.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area

        coverage = intersection_area / goal_area if goal_area > 0 else 0.0
        reward = np.clip(coverage, 0.0, 1.0)
        done = coverage > self.success_threshold
        return reward, done

    @staticmethod
    def _pymunk_to_shapely(body: pymunk.Body, shapes: list) -> sg.MultiPolygon:
        """Converts a Pymunk body's shapes to a Shapely MultiPolygon."""
        geoms = [
            sg.Polygon([body.local_to_world(v) for v in shape.get_vertices()])
            for shape in shapes
            if isinstance(shape, pymunk.Poly)
        ]
        return sg.MultiPolygon(geoms)


# ============================== EXAMPLE USAGE ==============================
if __name__ == "__main__":
    # You can now tune the physics by changing these parameters
    env = PushTEnv(
        render_size=256,
        damping=0.1,  # Higher value = more "drag", less sliding
        block_friction=0.8,  # Higher value = less slippery block
    )

    obs, info = env.reset(seed=42)
    print("Initial Observation Shape:", obs.shape)

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        env.render_mode = "human"
        env.render()

        if i % 10 == 0:
            print(f"Step: {i}, Reward: {reward:.3f}, Done: {done}")

        if done:
            print("Goal reached! Resetting.")
            obs, info = env.reset()

    env.close()
    print("Environment closed.")
