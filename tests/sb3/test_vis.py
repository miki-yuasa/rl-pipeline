from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.sb3.callback import VideoRecorderCallback


def _dummy_model(env: gym.Env[Any, Any]) -> BaseAlgorithm:
    """Creates a mock policy algorithm adhering to BaseAlgorithm's spec."""
    model = mock.create_autospec(BaseAlgorithm, instance=True, spec_set=True)
    model.predict.side_effect = lambda obs, **kwargs: (
        env.action_space.sample(),
        None,
    )
    return model


def test_video_recorder_callback_records_checkpoints(
    tmp_path: pathlib.Path,
) -> None:
    env = TimeLimit(
        gym.make("CartPole-v1", render_mode="rgb_array"), max_episode_steps=2
    )
    model = _dummy_model(env)

    callback = VideoRecorderCallback(
        eval_env=env,
        render_freq=10,
        save_dir=str(tmp_path),
        name_prefix="test_model",
    )
    callback.init_callback(model)

    # First checkpoint recording.
    callback.n_calls = 10
    callback.num_timesteps = 100
    callback._on_step()
    assert (tmp_path / "test_model_100_steps.gif").exists()

    # Subsequent checkpoint verifies the environment is not prematurely closed.
    callback.n_calls = 20
    callback.num_timesteps = 200
    callback._on_step()
    assert (tmp_path / "test_model_200_steps.gif").exists()

    # Environment is closed upon training completion.
    callback._on_training_end()


def test_video_recorder_callback_with_custom_player(
    tmp_path: pathlib.Path,
) -> None:
    env = TimeLimit(
        gym.make("CartPole-v1", render_mode="rgb_array"), max_episode_steps=2
    )
    model = _dummy_model(env)
    custom_player_calls: list[dict[str, Any]] = []

    def mock_player(
        eval_env: gym.Env[Any, Any],
        algo: BaseAlgorithm,
        save_path: str,
        verbose: bool = False,
        close_env: bool = False,
    ) -> None:
        custom_player_calls.append(
            {"save_path": save_path, "close_env": close_env}
        )

    callback = VideoRecorderCallback(
        eval_env=env,
        render_freq=5,
        save_dir=str(tmp_path),
        name_prefix="custom_model",
        custom_player=mock_player,
    )
    callback.init_callback(model)

    callback.n_calls = 5
    callback.num_timesteps = 50
    callback._on_step()

    assert len(custom_player_calls) == 1
    assert custom_player_calls[0]["close_env"] is False
    assert "custom_model_50_steps.gif" in custom_player_calls[0]["save_path"]
    callback._on_training_end()
