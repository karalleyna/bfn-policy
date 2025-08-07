from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

from env_configs.pusht import PushTConfig
from simulators.pusht import PushTPhysicsSimulator


class PushTRenderer:
    """Handles all PyGame rendering for the Push-T task."""

    def __init__(self, config: PushTConfig):
        """Initializes the renderer.

        Args:
          config: The configuration object for the environment.
        """
        self._config = config
        self.window = None
        self.clock = None
        self._goal_verts = self._precompute_goal_vertices()

    def _initialize_pygame(self):
        """Initializes PyGame and the display window if not already done."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self._config.window_size, self._config.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _precompute_goal_vertices(self) -> List[List[Tuple[float, float]]]:
        """Pre-computes the vertices for the goal region for efficient rendering.

        This avoids recalculating the goal's geometry on every render frame.

        Returns:
            A list of polygons, where each polygon is a list of vertices.
        """
        # Create a temporary static body to represent the goal pose.
        goal_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        goal_body.position = self._config.goal_pose[:2]
        goal_body.angle = self._config.goal_pose[2]

        # Create a temporary block to get the shape geometry.
        temp_sim = PushTPhysicsSimulator(self._config)
        block_shapes = temp_sim.block.shapes

        # Transform the block's local vertices to the goal's world coordinates.
        all_verts = []
        for shape in block_shapes:
            verts = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            all_verts.append(verts)
        return all_verts

    def _draw_goal(self, surface: pygame.Surface):
        """Draws the pre-computed goal area onto the canvas."""
        for verts in self._goal_verts:
            pygame.draw.polygon(surface, pygame.Color("LightGreen"), verts)

    def render_frame(
        self, sim_state: Dict[str, Any], mode: str
    ) -> Optional[np.ndarray]:
        """Renders a single frame of the environment.

        Args:
          sim_state: The current state of the physics simulation.
          mode: The rendering mode ('human' or 'rgb_array').

        Returns:
          An RGB array of the rendered frame if mode is 'rgb_array', else None.
        """
        self._initialize_pygame()
        canvas = pygame.Surface((self._config.window_size, self._config.window_size))
        canvas.fill(pygame.Color("white"))

        # Draw the goal area first, so it's in the background.
        self._draw_goal(canvas)

        # Draw the simulation objects (agent, block, walls).
        draw_options = pymunk.pygame_util.DrawOptions(canvas)
        sim_state["space"].debug_draw(draw_options)

        if mode == "human":
            # Blit the canvas to the display window and update.
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(30)  # Limit human-mode rendering to 30 FPS.
            return None

        # For 'rgb_array' mode, convert the PyGame surface to a NumPy array.
        img_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

        # Resize to the desired observation size.
        return cv2.resize(
            img_array, (self._config.RENDER_SIZE, self._config.RENDER_SIZE)
        )

    def close(self):
        """Closes the PyGame window and quits PyGame."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
