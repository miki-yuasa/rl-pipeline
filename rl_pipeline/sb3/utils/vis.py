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
) -> None:
    """Records an evaluation episode replay and saves it as an animation.

    Args:
        demo_env: Evaluation environment to rollout.
        model: Policy algorithm used to sample actions.
        animation_save_path: Filepath where the animation should be saved.
        verbose: Whether to print rollout transition details.
        close_env: Whether to close the environment upon recording completion.
    """
    obs, _ = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    frames = [demo_env.render()]
    rewards: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs)  # type: ignore
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
        rewards.append(reward)  # type: ignore
        frame = demo_env.render()
        frames.append(frame)

    if close_env:
        demo_env.close()
    if verbose:
        print(f" - Total reward: {sum(rewards)}")

    save_path = Path(animation_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, fps=10, dpi=300, loop=10)  # type: ignore
    if verbose:
        print(f" - Replay saved to {animation_save_path}")
