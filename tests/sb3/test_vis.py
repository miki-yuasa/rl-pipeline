from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import gymnasium as gym
from absl.testing import absltest
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.sb3.callback import VideoRecorderCallback


def _dummy_model(env: gym.Env[Any, Any]) -> BaseAlgorithm:
    model = mock.create_autospec(BaseAlgorithm, instance=True, spec_set=True)
    model.predict.side_effect = lambda obs, **kwargs: (
        env.action_space.sample(),
        None,
    )
    return model


class VideoRecorderCallbackTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir = Path(self.create_tempdir().full_path)

    def test_video_recorder_callback_records_checkpoints(self) -> None:
        env = TimeLimit(
            gym.make("CartPole-v1", render_mode="rgb_array"),
            max_episode_steps=2,
        )
        model = _dummy_model(env)

        callback = VideoRecorderCallback(
            eval_env=env,
            render_freq=10,
            save_dir=str(self.temp_dir),
            name_prefix="test_model",
        )
        callback.init_callback(model)

        callback.n_calls = 10
        callback.num_timesteps = 100
        callback._on_step()
        self.assertTrue((self.temp_dir / "test_model_100_steps.gif").exists())

        callback.n_calls = 20
        callback.num_timesteps = 200
        callback._on_step()
        self.assertTrue((self.temp_dir / "test_model_200_steps.gif").exists())

        callback._on_training_end()

    def test_video_recorder_callback_with_custom_player(self) -> None:
        env = TimeLimit(
            gym.make("CartPole-v1", render_mode="rgb_array"),
            max_episode_steps=2,
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
            save_dir=str(self.temp_dir),
            name_prefix="custom_model",
            custom_player=mock_player,
        )
        callback.init_callback(model)

        callback.n_calls = 5
        callback.num_timesteps = 50
        callback._on_step()

        self.assertLen(custom_player_calls, 1)
        self.assertFalse(custom_player_calls[0]["close_env"])
        self.assertIn(
            "custom_model_50_steps.gif", custom_player_calls[0]["save_path"]
        )
        callback._on_training_end()


if __name__ == "__main__":
    absltest.main()
