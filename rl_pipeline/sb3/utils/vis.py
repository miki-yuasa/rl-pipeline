from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Any

import imageio
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm


def record_replay(
    demo_env: Env[Any, Any],
    model: BaseAlgorithm,
    animation_save_path: str,
    verbose: bool = True,
    close_env: bool = True,
    fps: int = 10,
) -> None:
    """Records an evaluation episode replay and saves it as an animation.

    Args:
        demo_env: Evaluation environment to rollout.
        model: Policy algorithm used to sample actions.
        animation_save_path: Filepath where the animation should be saved.
        verbose: Whether to print rollout transition details.
        close_env: Whether to close the environment upon recording completion.
        fps: Frames per second for the saved animation.
    """
    obs, _ = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    first_frame = demo_env.render()
    frames: list[Any] = []
    if isinstance(first_frame, list):
        frames.extend(first_frame)
    elif first_frame is not None:
        frames.append(first_frame)

    rewards: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs)  # type: ignore[assignment]
        obs, reward, terminated, truncated, info = demo_env.step(action)
        if verbose:
            print(f"Step {len(rewards) + 1}:")
            print(" - Action taken:")
            pprint(action)
            print(
                f" - Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Success: {info.get('is_success', 'N/A')}"
            )
            print(" - Observation: ")
            pprint(obs)
            print(" - Info: ")
            pprint(info)
        rewards.append(float(reward))
        frame = demo_env.render()
        if isinstance(frame, list):
            frames.extend(frame)
        elif frame is not None:
            frames.append(frame)

    if close_env:
        demo_env.close()
    if verbose:
        print(f" - Total reward: {sum(rewards)}")

    save_path = Path(animation_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        duration_ms = 1000.0 / max(1, fps)
        imageio.mimsave(save_path, frames, duration=duration_ms, loop=0)  # type: ignore[call-overload]
    if verbose:
        print(f" - Replay saved to {animation_save_path}")
