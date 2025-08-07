"""
Renderer for the Push-T environment using PyGame.

This module handles window creation, drawing all environment elements,
and converting the PyGame surface to a NumPy array for observations.
"""
import cv2
import numpy as np
import pygame
import pymunk.pygame_util

from .config import PushTConfig


class PushTRenderer:
    """Manages all PyGame-based rendering for the Push-T environment."""

    def __init__(self, config: PushTConfig):
        self.config = config
        self.window = None
        self.clock = None
        self.screen = None

    def render_frame(
        self, sim_state: dict, mode: str, latest_action: np.ndarray
    ) -> np.ndarray:
        """
        Renders a single frame of the environment.

        Args:
            sim_state: A dictionary containing the Pymunk bodies to render.
            mode: The rendering mode, either "human" or "rgb_array".
            latest_action: The last action taken, to be drawn if configured.

        Returns:
            A NumPy array of the rendered image.
        """
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.config.window_size, self.config.window_size)
            )
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        # Create the drawing surface
        canvas = pygame.Surface((self.config.window_size, self.config.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas
        draw_options = pymunk.pygame_util.DrawOptions(canvas)

        # Draw goal pose
        self._draw_goal(sim_state["block_body"])

        # Draw all simulation objects
        sim_state["agent_body"].space.debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.config.control_hz)

        # Convert to numpy array and resize
        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.config.render_size, self.config.render_size))

        # Draw action marker if enabled
        if self.config.render_action and latest_action is not None:
            self._draw_action_marker(img, latest_action)

        return img

    def _draw_goal(self, block_body: pymunk.Body):
        """Helper to draw the goal region."""
        goal_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        goal_body.position = self.config.goal_pose[:2]
        goal_body.angle = self.config.goal_pose[2]

        for shape in block_body.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), self.screen)
                for v in shape.get_vertices()
            ]
            pygame.draw.polygon(self.screen, self.config.goal_color, goal_points)

    def _draw_action_marker(self, image: np.ndarray, action: np.ndarray):
        """Draws a marker for the last executed action."""
        coord = (action / self.config.window_size * self.config.render_size).astype(
            np.int32
        )
        marker_size = int(8 / 96 * self.config.render_size)
        thickness = int(1 / 96 * self.config.render_size)
        cv2.drawMarker(
            image,
            tuple(coord),
            self.config.action_color,
            cv2.MARKER_CROSS,
            marker_size,
            thickness,
        )

    def close(self):
        """Closes the PyGame window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
