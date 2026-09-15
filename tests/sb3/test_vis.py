from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.sb3.callback import VideoRecorderCallback
from rl_pipeline.sb3.utils.vis import record_replay


def _dummy_model(env: gym.Env) -> BaseAlgorithm:
    model = MagicMock(spec=BaseAlgorithm)
    model.predict.side_effect = lambda obs: (env.action_space.sample(), None)
    return model


def test_record_replay_keep_env_open(tmp_path: pathlib.Path) -> None:
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    model = _dummy_model(env)
    save_file = tmp_path / "replay_1.gif"

    record_replay(env, model, str(save_file), verbose=False, close_env=False)
    assert save_file.exists()

    # The environment should still be usable without re-creating it
    save_file_2 = tmp_path / "replay_2.gif"
    record_replay(env, model, str(save_file_2), verbose=False, close_env=True)
    assert save_file_2.exists()


def test_video_recorder_callback_lifecycle(tmp_path: pathlib.Path) -> None:
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    model = _dummy_model(env)

    callback = VideoRecorderCallback(
        eval_env=env,
        render_freq=1,
        save_dir=str(tmp_path),
        name_prefix="test_model",
    )
    callback.init_callback(model)

    callback.n_calls = 1
    callback.num_timesteps = 100
    callback._on_step()

    expected_file = tmp_path / "test_model_100_steps.gif"
    assert expected_file.exists()

    callback._on_training_end()
