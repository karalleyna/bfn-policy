"""
Main Gymnasium environment file for the Push-T task.

This file defines the `PushTEnv` and `PushTKeypointsEnv` classes, which
orchestrate the simulator, renderer, and other components to create a
fully functional Gym environment.
"""
import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pymunk
import shapely.geometry as sg

from .config import PushTConfig
from .keypoint_manager import PymunkKeypointManager
from .renderer import PushTRenderer
from .simulator import PushTPhysicsSimulator


class PushTEnv(gym.Env):
    """The main Push-T environment, orchestrating simulation and rendering."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    reward_range = (0.0, 1.0)

    def __init__(self, config: Optional[PushTConfig] = None, **kwargs):
        super().__init__()

        # Separate kwargs for the config and for the env.
        config_fields = {f.name for f in dataclasses.fields(PushTConfig)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        env_kwargs = {k: v for k, v in kwargs.items() if k not in config_fields}

        # Instantiate the configuration.
        base_config_dict = {}
        if config is not None and hasattr(config, "to_dict"):
            base_config_dict = config.to_dict()
        base_config_dict.update(config_kwargs)
        self.config = PushTConfig(**base_config_dict)

        # Handle environment-specific arguments.
        self.render_mode = env_kwargs.get("render_mode", None)

        # Initialize components with the final config.
        self.simulator = PushTPhysicsSimulator(self.config)
        self.renderer = PushTRenderer(self.config)

        # Create observation space directly in the constructor.
        ws = self.config.window_size
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([ws, ws, ws, ws, 2 * np.pi], dtype=np.float32),
        )
        self.action_space = self._create_action_space()

        self._goal_geom = self._precompute_goal_geometry()
        self.latest_action = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        state = self.np_random.uniform(
            low=[50, 50, 100, 100, -np.pi], high=[450, 450, 400, 400, np.pi], size=(5,)
        )
        self.simulator.set_state(state)
        self.latest_action = None
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.latest_action = action.copy()
        self.simulator.step(action)

        obs = self._get_obs()
        reward, done = self._calculate_reward()
        info = self._get_info()
        truncated = False  # This environment does not have a time limit

        return obs, reward, done, truncated, info

    def render(self) -> np.ndarray:
        if self.render_mode is None:
            raise ValueError(
                "Render mode is not set. Please set `env.render_mode` before calling `render()`."
            )
        return self.renderer.render_frame(
            sim_state=self.simulator.get_state_dict(),
            mode=self.render_mode,
            latest_action=self.latest_action,
        )

    def close(self):
        self.renderer.close()

    def _create_action_space(self) -> gym.spaces.Box:
        ws = self.config.window_size
        return gym.spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([ws, ws], dtype=np.float32),
        )

    def _get_obs(self) -> np.ndarray:
        agent_pos = self.simulator.agent.position
        block_pos = self.simulator.block.position
        block_angle = self.simulator.block.angle % (2 * np.pi)
        return np.array(
            list(agent_pos) + list(block_pos) + [block_angle], dtype=np.float32
        )

    def _get_info(self) -> Dict[str, Any]:
        return {
            "block_pose": np.array(
                list(self.simulator.block.position) + [self.simulator.block.angle]
            )
        }

    def _calculate_reward(self) -> Tuple[float, bool]:
        # FIX: Apply a buffer of 0 to resolve potential invalid geometries
        # arising from the T-block's construction (overlapping shapes).
        block_geom = self._pymunk_to_shapely(
            self.simulator.block, self.simulator.block.shapes
        ).buffer(0)

        union_geom = self._goal_geom.union(block_geom)
        iou = (
            self._goal_geom.intersection(block_geom).area / union_geom.area
            if union_geom.area > 0
            else 0.0
        )

        reward = np.clip(iou / self.config.success_threshold, 0, 1)
        done = iou > self.config.success_threshold
        return reward, done

    def _precompute_goal_geometry(self) -> sg.MultiPolygon:
        goal_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        goal_body.position = self.config.goal_pose[:2]
        goal_body.angle = self.config.goal_pose[2]

        # FIX: Also buffer the goal geometry to ensure it is valid.
        geom = self._pymunk_to_shapely(goal_body, self.simulator.block.shapes).buffer(0)
        return geom

    @staticmethod
    def _pymunk_to_shapely(
        body: pymunk.Body, shapes: List[pymunk.Shape]
    ) -> sg.MultiPolygon:
        geoms = []
        for shape in shapes:
            if isinstance(shape, pymunk.Poly):
                verts = [body.local_to_world(v) for v in shape.get_vertices()]
                geoms.append(sg.Polygon(verts))
        return sg.MultiPolygon(geoms)


class PushTKeypointsEnv(PushTEnv):
    """An extension of PushTEnv that uses keypoints for observations."""

    def __init__(
        self, keypoint_visible_rate: float = 1.0, draw_keypoints: bool = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.keypoint_manager = PymunkKeypointManager.create_from_config(self.config)
        self.keypoint_visible_rate = keypoint_visible_rate
        self.draw_keypoints = draw_keypoints
        self._last_global_kp_map = {}
        self._last_kp_mask = {}

        self.observation_space = self._create_observation_space()

    def _create_observation_space(self) -> gym.spaces.Box:
        n_block_kps = len(self.keypoint_manager.local_keypoint_map["block"])
        n_agent_kps = len(self.keypoint_manager.local_keypoint_map["agent"])
        n_total_kps = n_block_kps + n_agent_kps
        obs_dim = (n_total_kps * 2) + n_total_kps  # kps + mask

        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones_like(low) * self.config.window_size
        high[n_total_kps * 2 :] = 1.0
        return gym.spaces.Box(low=low, high=high)

    def _get_obs(self) -> np.ndarray:
        pose_map = {"block": self.simulator.block, "agent": self.simulator.agent}
        self._last_global_kp_map = self.keypoint_manager.get_keypoints_global(pose_map)

        kp_names = sorted(self._last_global_kp_map.keys())
        kps_list = [self._last_global_kp_map[name] for name in kp_names]

        mask_list = []
        for i, name in enumerate(kp_names):
            mask = self.np_random.random(len(kps_list[i])) < self.keypoint_visible_rate
            self._last_kp_mask[name] = mask
            mask_list.append(mask)

        flat_kps = np.concatenate([kp.flatten() for kp in kps_list]).astype(np.float32)
        flat_mask = np.concatenate(mask_list).astype(np.float32)
        return np.concatenate([flat_kps, flat_mask])

    def render(self) -> np.ndarray:
        img = super().render()
        if self.draw_keypoints:
            visible_kp_map = {}
            for name, kps in self._last_global_kp_map.items():
                if name in self._last_kp_mask:
                    visible_kp_map[name] = kps[self._last_kp_mask[name]]

            self.keypoint_manager.draw_keypoints_on_numpy_array(
                image=img,
                global_kp_map=visible_kp_map,
                pymunk_space_size=self.config.window_size,
            )
        return img
