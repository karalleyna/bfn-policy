"""
A manager for defining, transforming, and rendering keypoints on Pymunk objects.
"""
from typing import Dict, Optional

import cv2
import numpy as np
import pymunk


class PymunkKeypointManager:
    """Manages keypoint definitions, transformations, and rendering."""

    def __init__(
        self,
        local_keypoint_map: Dict[str, np.ndarray],
        color_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.local_keypoint_map = {
            name: np.array(kps) for name, kps in local_keypoint_map.items()
        }
        self.color_map = (
            color_map if color_map is not None else self._generate_default_color_map()
        )

    def _generate_default_color_map(self) -> Dict[str, np.ndarray]:
        """Creates a default color map if none is provided."""
        return {name: np.array([255, 0, 0]) for name in self.local_keypoint_map.keys()}

    def get_keypoints_global(
        self, pose_map: Dict[str, pymunk.Body]
    ) -> Dict[str, np.ndarray]:
        """Transforms local keypoints to global (world) coordinates."""
        global_kp_map = {}
        for name, body in pose_map.items():
            if name in self.local_keypoint_map:
                local_kps = self.local_keypoint_map[name]
                global_kps = [body.local_to_world(tuple(kp)) for kp in local_kps]
                global_kp_map[name] = np.array(global_kps, dtype=np.float32)
        return global_kp_map

    def draw_keypoints_on_numpy_array(
        self,
        image: np.ndarray,
        global_kp_map: Dict[str, np.ndarray],
        pymunk_space_size: int,
    ):
        """Draws keypoints onto a NumPy array image using OpenCV."""
        render_size = image.shape[0]
        radius = int(3 / 96 * render_size)

        for name, kps in global_kp_map.items():
            if name in self.color_map and len(kps) > 0:
                color_bgr = tuple(int(c) for c in self.color_map[name][::-1])
                for kp in kps:
                    coord = (np.array(kp) / pymunk_space_size * render_size).astype(
                        np.int32
                    )
                    cv2.circle(image, tuple(coord), radius, color_bgr, thickness=-1)

    @classmethod
    def create_from_config(cls, config):
        """Factory method to create a keypoint manager from a config."""
        scale = config.block_scale
        # Keypoints for the T-block, based on its new construction
        block_keypoints = np.array(
            [
                (0, 0),
                (-2 * scale, 0.5 * scale),
                (2 * scale, 0.5 * scale),
                (-2 * scale, -0.5 * scale),
                (2 * scale, -0.5 * scale),
                (0, 2.5 * scale),
                (0, 0.5 * scale),
            ],
            dtype=np.float32,
        )

        agent_keypoints = np.array([[0, 0]], dtype=np.float32)

        local_keypoint_map = {"block": block_keypoints, "agent": agent_keypoints}
        color_map = {"block": np.array([255, 0, 0]), "agent": np.array([0, 0, 255])}
        return cls(local_keypoint_map=local_keypoint_map, color_map=color_map)
