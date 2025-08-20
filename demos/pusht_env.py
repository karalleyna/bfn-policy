from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pygame
import pymunk
import zarr
from pymunk.vec2d import Vec2d

from datasets.utils.replay_buffer import ReplayBuffer

# Import the new keypoints environment
from environments.pusht.environment import PushTKeypointsEnv


def get_teleop_action(
    env: PushTKeypointsEnv, is_teleop_active: bool
) -> Tuple[np.ndarray, bool]:
    """
    Gets a teleoperation action from the mouse position.
    """
    action = None
    if env.renderer.window is not None:
        mouse_pos_pixels = pygame.mouse.get_pos()
        mouse_pos_pymunk = pymunk.pygame_util.from_pygame(
            Vec2d(*mouse_pos_pixels), env.renderer.window
        )
        agent_pos = env.simulator.agent.position
        dist_to_agent = (mouse_pos_pymunk - agent_pos).length

        if not is_teleop_active and dist_to_agent < env.config.agent_radius:
            is_teleop_active = True

        if is_teleop_active:
            action = np.array(mouse_pos_pymunk, dtype=np.float32)

    return action, is_teleop_active


@click.command(
    help="""
    Collect human demonstrations for the Push-T task using the Keypoints Environment.
    
    \b
    Usage:
    python my_pusht_demo.py -o ./data/pusht_keypoints_demo.zarr

    \b
    Controls:
    - Hover mouse near the blue agent to start controlling it.
    - Push the gray T-block into the green goal area.
    - Press 'R' to restart the episode.
    - Press 'Q' to quit the application.
    - Hold 'SPACE' to pause the data recording.
    """
)
@click.option("-o", "--output", required=True, help="Path to the output Zarr file.")
@click.option(
    "-rs",
    "--render_size",
    default=96,
    type=int,
    help="Size of the rendered image observation.",
)
@click.option(
    "-hz", "--control_hz", default=10, type=int, help="Control frequency in Hz."
)
def main(output, render_size, control_hz):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    replay_buffer = ReplayBuffer.from_path(output_path, mode="a")
    # =========================================================================
    # Instantiate the new PushTKeypointsEnv
    # We set draw_keypoints=True to visualize them during data collection.
    # =========================================================================
    env = PushTKeypointsEnv(
        render_mode="human", render_size=render_size, draw_keypoints=True
    )

    print("Instructions: Control the blue agent with your mouse.")
    print("Push the gray T-block into the green area.")
    print("Press 'R' for a new episode, 'Q' to quit.")

    is_running = True
    while is_running:
        episode_data = []
        seed = replay_buffer.n_episodes
        print(f"--- Starting new episode (seed: {seed}) ---")

        obs, info = env.reset(seed=seed)
        img = env.render()

        is_teleop_active = False
        is_paused = False
        is_retry = False

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        is_running = False
                        break
                    if event.key == pygame.K_r:
                        is_retry = True
                    if event.key == pygame.K_SPACE:
                        is_paused = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        is_paused = False

            if not is_running or is_retry:
                break

            if is_paused:
                pygame.time.wait(10)
                continue

            action, is_teleop_active = get_teleop_action(env, is_teleop_active)

            # Ensure we always step with a valid action
            action_to_step = (
                action if action is not None else np.array(env.simulator.agent.position)
            )
            next_obs, reward, done, truncated, info = env.step(action_to_step)
            next_img = env.render()

            # Only record data when the mouse is controlling the agent
            if is_teleop_active and action is not None:
                data_step = {
                    "obs": obs,
                    "img": img,
                    "action": action,
                }
                episode_data.append(data_step)

            obs, img = next_obs, next_img

            if done:
                print(f"Success! Episode {seed} finished.")
                break

        if not is_retry and len(episode_data) > 0:
            keys = episode_data[0].keys()
            episode_to_save = {k: np.stack([d[k] for d in episode_data]) for k in keys}
            replay_buffer.add_episode(episode_to_save)
            print(f"Episode {seed} saved to {output_path} ({len(episode_data)} steps).")
        elif not is_retry:
            print(f"Episode {seed} discarded (no steps recorded).")
        else:
            print(f"Episode {seed} discarded (retry).")

    env.close()


if __name__ == "__main__":
    main()
