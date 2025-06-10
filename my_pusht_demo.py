from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pygame
import pymunk
import zarr
from pymunk.vec2d import Vec2d

from my_datasets.utils.replay_buffer import ReplayBuffer
from my_environments.pusht import PushTEnv


def get_teleop_action(env: PushTEnv, is_teleop_active: bool) -> Tuple[np.ndarray, bool]:
    """
    Gets a teleoperation action from the mouse position.

    Args:
        env: The PushTEnv instance.
        is_teleop_active: A flag indicating if teleoperation has started.

    Returns:
        A tuple of (action, is_teleop_active). Action is None if the
        mouse is not controlling the agent.
    """
    action = None
    mouse_pos_pixels = pygame.mouse.get_pos()

    if env.renderer.window is not None:
        mouse_pos_pymunk = pymunk.pygame_util.from_pygame(
            Vec2d(*mouse_pos_pixels), env.renderer.window
        )
        agent_pos = env.simulator.agent.position
        dist_to_agent = (mouse_pos_pymunk - agent_pos).length

        if not is_teleop_active and dist_to_agent < 15:  # agent radius
            is_teleop_active = True

        if is_teleop_active:
            action = np.array(mouse_pos_pymunk, dtype=np.float32)

    return action, is_teleop_active


# =========================== MAIN DEMO COLLECTION SCRIPT ===========================


@click.command(
    help="""
    Collect human demonstrations for the Push-T task.
    
    \b
    Usage:
    python collect_demo.py -o ./data/pusht_demo.zarr

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
    env = PushTEnv(render_size=render_size)
    env.render_mode = "human"

    print("Instructions: Control the blue agent with your mouse.")
    print("Push the gray T-block into the green area.")
    print("Press 'R' for a new episode, 'Q' to quit.")

    while True:
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
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        env.close()
                        return
                    if event.key == pygame.K_r:
                        is_retry = True
                    if event.key == pygame.K_SPACE:
                        is_paused = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        is_paused = False

            if is_retry:
                break

            if is_paused:
                pygame.time.wait(10)
                continue

            action, is_teleop_active = get_teleop_action(env, is_teleop_active)

            action_to_step = (
                action if action is not None else env.simulator.agent.position
            )
            next_obs, reward, done, truncated, info = env.step(action_to_step)
            next_img = env.render()

            if is_teleop_active:
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
        else:
            print(f"Episode {seed} discarded.")


if __name__ == "__main__":
    main()
