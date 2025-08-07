"""A manager for defining, transforming, and rendering keypoints on Pymunk objects.

This module provides a reusable class, `PymunkKeypointManager`, for handling
keypoint operations in a Pymunk-based environment. It is designed to be
decoupled from the main environment logic, promoting modularity and reusability.
"""

from typing import Dict, Optional

import cv2
import numpy as np
import pygame
import pymunk


class PymunkKeypointManager:
    """Manages keypoint definitions, transformations, and rendering."""

    def __init__(
        self,
        local_keypoint_map: Dict[str, np.ndarray],
        color_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Initializes the keypoint manager.

        Args:
            local_keypoint_map: A dictionary mapping object names (e.g., 'block',
                'agent') to their keypoints defined in local coordinates. The
                shape of each value should be (n_keypoints, 2).
            color_map: An optional dictionary mapping object names to the RGB
                color used for rendering their keypoints.
        """
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
        """Transforms local keypoints to global (world) coordinates.

        Args:
            pose_map: A dictionary mapping object names to their corresponding
                Pymunk Body objects, which contain the global pose information.

        Returns:
            A dictionary mapping object names to their keypoints in global
            world coordinates.
        """
        global_kp_map = {}
        for name, body in pose_map.items():
            if name in self.local_keypoint_map:
                local_kps = self.local_keypoint_map[name]
                global_kps = [body.local_to_world(tuple(kp)) for kp in local_kps]
                global_kp_map[name] = np.array(global_kps, dtype=np.float32)
        return global_kp_map

    def draw_keypoints_on_surface(
        self,
        surface: pygame.Surface,
        screen_to_pymunk_transform: pymunk.Transform,
        global_kp_map: Dict[str, np.ndarray],
        radius: int = 3,
    ):
        """Draws keypoints onto a PyGame surface.

        Args:
            surface: The PyGame surface to draw on.
            screen_to_pymunk_transform: The transformation required to convert
                Pymunk world coordinates to PyGame screen coordinates.
            global_kp_map: A map of object names to their global keypoints.
            radius: The radius of the circles used to draw keypoints.
        """
        for name, kps in global_kp_map.items():
            if name in self.color_map:
                color = self.color_map[name]
                for kp in kps:
                    # Convert Pymunk coordinates to PyGame screen coordinates
                    screen_pos = screen_to_pymunk_transform.transform_point(kp)
                    pygame.draw.circle(
                        surface, color, tuple(map(int, screen_pos)), radius
                    )

    def draw_keypoints_on_numpy_array(
        self,
        image: np.ndarray,
        global_kp_map: Dict[str, np.ndarray],
        pymunk_space_size: int,
        radius: int = 3,
        thickness: int = -1,
    ):
        """Draws keypoints onto a NumPy array image using OpenCV.

        Args:
            image: The image (NumPy array) to draw on.
            global_kp_map: A map of object names to their global keypoints.
            pymunk_space_size: The size of the Pymunk simulation space (e.g., 512).
            radius: The radius of the keypoint circles in pixels.
            thickness: The thickness of the circle outline. -1 fills the circle.
        """
        render_size = image.shape[0]  # Assuming a square image
        for name, kps in global_kp_map.items():
            if name in self.color_map and len(kps) > 0:
                # Pygame uses RGB, OpenCV uses BGR. Convert color.
                color_bgr = tuple(int(c) for c in self.color_map[name][::-1])
                for kp in kps:
                    # Transform from Pymunk world coordinates to image pixel coordinates
                    coord = (np.array(kp) / pymunk_space_size * render_size).astype(
                        np.int32
                    )
                    cv2.circle(image, tuple(coord), radius, color_bgr, thickness)

    @classmethod
    def create_from_pusht_env(cls, env):
        """A factory method to create a keypoint manager from a PushTEnv instance.

        This demonstrates how to define keypoints for the specific geometry
        of the Push-T environment.

        Args:
            env: An instance of the PushTEnv.

        Returns:
            A PymunkKeypointManager instance with predefined keypoints for Push-T.
        """
        # Define keypoints for the T-shaped block in its local coordinate frame
        # These values are chosen to mark corners and centers of the T-block parts.
        block_keypoints = (
            np.array(
                [[0, 0], [-60, 15], [-60, -15], [60, 15], [60, -15], [0, 75], [0, -75]],
                dtype=np.float32,
            )
            / 2.0
        )  # Scale to match the tee definition in the new env

        # Define keypoints for the circular agent
        agent_keypoints = np.array(
            [[0, 0]],  # Center
            dtype=np.float32,
        )

        local_keypoint_map = {"block": block_keypoints, "agent": agent_keypoints}
        color_map = {
            "block": np.array([255, 0, 0]),  # Red
            "agent": np.array([0, 0, 255]),  # Blue
        }
        return cls(local_keypoint_map=local_keypoint_map, color_map=color_map)
