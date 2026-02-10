import os
from pprint import pprint

import imageio
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm


def record_replay(
    demo_env: Env, model: BaseAlgorithm, animation_save_path: str, verbose: bool = True
) -> None:
    obs, _ = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    frames = [demo_env.render()]
    rewards: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs)  # type: ignore
        # Ensure action is a numpy int64 scalar
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

    demo_env.close()
    if verbose:
        print(f" - Total reward: {sum(rewards)}")

    os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
    imageio.mimsave(animation_save_path, frames, fps=10, dpi=300, loop=10)  # type: ignore
    if verbose:
        print(f" - Replay saved to {animation_save_path}")
