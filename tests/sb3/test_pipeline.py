from __future__ import annotations

from pathlib import Path
from unittest import mock

import gymnasium as gym
from absl.testing import absltest
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader
from rl_pipeline.sb3.callback import (
    CheckpointCallbackConfig,
    EvalCallbackConfig,
    SuccessEvalCallback,
)
from rl_pipeline.sb3.config import (
    ArbitraryCallbackConfig,
    SB3CallbackConfig,
    SB3PipelineConfig,
)
from rl_pipeline.sb3.pipeline import init_callback


class _TestArbitraryCallback(BaseCallback):
    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag

    def _on_step(self) -> bool:
        return True


class SB3PipelineTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir = Path(self.create_tempdir().full_path)
        self.config: SB3PipelineConfig = SB3PipelineConfigReader.from_yaml(
            "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
        ).to_config()
        # Point paths to isolated temp directory and disable external experiment managers
        self.config.save_config.model_save_path = str(
            self.temp_dir / "models" / "final.zip"
        )
        self.config.save_config.best_model_save_path = str(
            self.temp_dir / "models" / "best.zip"
        )
        self.config.save_config.eval_save_dir = str(self.temp_dir / "eval")
        self.config.save_config.animation_save_path = str(
            self.temp_dir / "anim.gif"
        )
        self.config.experiment_manager_config = None

    def test_init_callback_appends_arbitrary_callbacks_in_order(self) -> None:
        callback_config = SB3CallbackConfig(
            eval_callback_config=EvalCallbackConfig(
                eval_freq=10, n_eval_episodes=2
            ),
            ckpt_callback_config=CheckpointCallbackConfig(
                save_freq=10,
                save_path=str(self.temp_dir / "ckpts"),
            ),
            arbitrary_callback_configs=[
                ArbitraryCallbackConfig(
                    callback_class=_TestArbitraryCallback,
                    callback_kwargs={"tag": "alpha"},
                )
            ],
        )

        eval_env = DummyVecEnv(
            [lambda: gym.make("CartPole-v1", render_mode="rgb_array")]
        )
        video_env = gym.make("CartPole-v1", render_mode="rgb_array")

        callbacks = init_callback(
            eval_env=eval_env,
            video_env=video_env,
            callback_config=callback_config,
        )

        eval_env.close()
        video_env.close()

        self.assertLen(callbacks, 3)
        self.assertIsInstance(callbacks[0], SuccessEvalCallback)
        self.assertIsInstance(callbacks[1], CheckpointCallback)
        arbitrary_cb = callbacks[2]
        self.assertIsInstance(arbitrary_cb, _TestArbitraryCallback)
        assert isinstance(arbitrary_cb, _TestArbitraryCallback)
        self.assertEqual(arbitrary_cb.tag, "alpha")

    def test_pipeline_initialization(self) -> None:
        pipeline = SB3Pipeline(config=self.config, verbose=False)
        self.assertEqual(pipeline.config, self.config)

    def test_pipeline_train_orchestrates_lifecycle(self) -> None:
        pipeline = SB3Pipeline(config=self.config, verbose=False)

        mock_env = mock.create_autospec(VecEnv, instance=True, spec_set=True)
        mock_model = mock.create_autospec(
            BaseAlgorithm, instance=True, spec_set=True
        )

        self.enter_context(
            mock.patch.object(
                pipeline.env_loader, "vec_env", return_value=mock_env
            )
        )
        self.enter_context(
            mock.patch.object(
                pipeline.model_loader, "model", return_value=mock_model
            )
        )

        trained_model = pipeline.train()

        self.assertIs(trained_model, mock_model)
        mock_model.learn.assert_called_once()
        mock_model.save.assert_called_once()
        mock_env.close.assert_called_once()

    def test_pipeline_load_model_calls_model_loader(self) -> None:
        pipeline = SB3Pipeline(config=self.config, verbose=False)
        mock_model = mock.create_autospec(
            BaseAlgorithm, instance=True, spec_set=True
        )

        mock_load = self.enter_context(
            mock.patch.object(
                pipeline.model_loader, "load_model", return_value=mock_model
            )
        )

        loaded = pipeline.load_model("best")

        self.assertIs(loaded, mock_model)
        mock_load.assert_called_once_with(
            self.config.save_config.best_model_save_path,
            None,
            self.config.device,
        )

    def test_pipeline_evaluate_computes_stats(self) -> None:
        pipeline = SB3Pipeline(config=self.config, verbose=False)
        mock_model = mock.create_autospec(
            BaseAlgorithm, instance=True, spec_set=True
        )
        mock_env = mock.create_autospec(VecEnv, instance=True, spec_set=True)

        self.enter_context(
            mock.patch.object(
                pipeline.env_loader, "vec_env", return_value=mock_env
            )
        )
        self.enter_context(
            mock.patch(
                "rl_pipeline.sb3.pipeline.evaluate_policy",
                return_value=([10.0, 20.0], [5, 15]),
            )
        )

        eval_result = pipeline.evaluate(
            checkpoint=mock_model, save_to_file=False
        )

        self.assertEqual(eval_result.mean_reward, 15.0)
        self.assertEqual(eval_result.mean_episode_length, 10.0)

    def test_pipeline_record_replay_calls_player(self) -> None:
        pipeline = SB3Pipeline(config=self.config, verbose=False)
        mock_model = mock.create_autospec(
            BaseAlgorithm, instance=True, spec_set=True
        )
        mock_player = mock.MagicMock()

        pipeline.record_replay(
            mock_model, custom_player=mock_player, verbose=False
        )

        mock_player.assert_called_once()


if __name__ == "__main__":
    absltest.main()
