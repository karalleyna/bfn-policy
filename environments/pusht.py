"""
This file contains the PushTEnv from the diffusion_policy repository,
along with a custom PushTKeypointsEnv that inherits from it to provide
keypoint-based observations.
"""
import collections
from typing import Dict

import cv2
import gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gym import spaces
from pymunk.pygame_util import DrawOptions
from pymunk.vec2d import Vec2d


def pymunk_to_shapely(body, shapes):
    """
    Converts a Pymunk body's shapes into a single Shapely MultiPolygon.
    """
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            pass
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTEnv(gym.Env):
    """
    PushTEnv from the diffusion_policy repository.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=True,
        render_size=96,
        reset_to_state=None,
    ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512
        self.render_size = render_size
        self.sim_hz = 100
        self.k_p, self.k_v = 100, 20
        self.control_hz = self.metadata["video.frames_per_second"]
        self.legacy = legacy

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float64),
            high=np.array([ws, ws, ws, ws, np.pi * 2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action
        self.window = None
        self.clock = None
        self.screen = None
        self.space = None
        self.teleop = False
        self.latest_action = None
        self.reset_to_state = reset_to_state
        self._setup()

    def reset(self):
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        state = self.reset_to_state
        if state is None:
            state = np.array(
                [
                    self.np_random.integers(50, 450),
                    self.np_random.integers(50, 450),
                    self.np_random.integers(100, 400),
                    self.np_random.integers(100, 400),
                    self.np_random.uniform(low=-np.pi, high=np.pi),
                ]
            )
        self._set_state(state)
        return self._get_obs()

    def step(self, action):
        dt = 1.0 / self.sim_hz
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for _ in range(n_steps):
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                    Vec2d(0, 0) - self.agent.velocity
                )
                self.agent.velocity += acceleration * dt
                self.space.step(dt)

        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area if goal_area > 0 else 0
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold

        return self._get_obs(), reward, done, self._get_info()

    def render(self, mode="human"):
        return self._render_frame(mode)

    def _get_obs(self):
        return np.array(
            tuple(self.agent.position)
            + tuple(self.block.position)
            + (self.block.angle % (2 * np.pi),)
        )

    def _get_info(self):
        return {"block_pose": np.array(list(self.block.position) + [self.block.angle])}

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas
        draw_options = DrawOptions(canvas)

        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(
                    goal_body.local_to_world(v), draw_options.surface
                )
                for v in shape.get_vertices()
            ]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        self.space.debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))

        if self.render_action and (self.latest_action is not None):
            action = np.array(self.latest_action)
            coord = (action / self.window_size * self.render_size).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(
                img, tuple(coord), (255, 0, 0), cv2.MARKER_CROSS, marker_size, thickness
            )
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _set_state(self, state):
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        self.block.angle = rot_block
        self.block.position = pos_block
        self.space.step(1.0 / self.sim_hz)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0.1
        self.teleop = False

        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color("LightGreen")
        self.goal_pose = np.array([256, 256, np.pi / 4])
        self.success_threshold = 0.95

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color="LightSlateGray"):
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        self.space.add(body, shape1, shape2)
        return body

    def _get_goal_pose_body(self, pose):
        body = pymunk.Body(1, pymunk.moment_for_box(1, (50, 100)))
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
