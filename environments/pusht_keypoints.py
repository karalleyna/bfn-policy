import numpy as np
from gym import spaces

from environments.pusht import PushTEnv

# Assuming keypoint_manager.py is in a location accessible by Python's path
from utils.keypoint_manager import PymunkKeypointManager


class PushTKeypointsEnv(PushTEnv):
    """
    An extension of the diffusion_policy PushTEnv that uses keypoints for observations.
    """

    def __init__(
        self, keypoint_visible_rate: float = 1.0, draw_keypoints: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.keypoint_manager = PymunkKeypointManager.create_from_pusht_env(self)
        self.keypoint_visible_rate = keypoint_visible_rate
        self.draw_keypoints = draw_keypoints
        self._last_global_kp_map = {}
        self._last_kp_mask = {}

        n_block_kps = len(self.keypoint_manager.local_keypoint_map["block"])
        n_agent_kps = len(self.keypoint_manager.local_keypoint_map["agent"])
        n_total_kps = n_block_kps + n_agent_kps

        obs_dim = (n_total_kps * 2) + n_total_kps

        low = np.zeros((obs_dim,), dtype=np.float32)
        high = np.ones_like(low) * self.window_size
        high[n_total_kps * 2 :] = 1.0

        self.observation_space = spaces.Box(
            low=low, high=high, shape=(obs_dim,), dtype=np.float32
        )

    def _get_obs(self):
        pose_map = {"block": self.block, "agent": self.agent}
        global_kp_map = self.keypoint_manager.get_keypoints_global(pose_map)
        self._last_global_kp_map = global_kp_map

        # Use a consistent order to prevent bugs
        kp_names = sorted(global_kp_map.keys())  # ['agent', 'block']

        kps_list = [global_kp_map[name] for name in kp_names]

        mask_list = []
        for name in kp_names:
            n_kps = global_kp_map[name].shape[0]
            mask = self.np_random.random(size=n_kps) < self.keypoint_visible_rate
            mask_list.append(mask)
            # Store the mask for rendering
            self._last_kp_mask[name] = mask

        # Explicitly define dtypes to avoid inference issues
        flat_kps = np.concatenate([kp.flatten() for kp in kps_list]).astype(np.float32)
        flat_mask = np.concatenate(mask_list).astype(np.float32)

        return np.concatenate([flat_kps, flat_mask])

    def render(self, mode="human"):
        img = super().render(mode)
        if self.draw_keypoints:
            # Create a map of only the visible keypoints for drawing
            visible_kp_map = {}
            for name, kps in self._last_global_kp_map.items():
                if name in self._last_kp_mask:
                    visible_kp_map[name] = kps[self._last_kp_mask[name]]

            self.keypoint_manager.draw_keypoints_on_numpy_array(
                image=img,
                global_kp_map=visible_kp_map,
                pymunk_space_size=self.window_size,
                radius=int(3 / 96 * self.render_size),
            )
        return img
