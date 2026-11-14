from __future__ import annotations

from pathlib import Path
from unittest import mock

from absl.testing import absltest, parameterized
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfig
from rl_pipeline.sb3 import loader as sb3_loader
from rl_pipeline.sb3 import pipeline as sb3_pipeline


class SB3PipelineTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir = Path(self.create_tempdir().full_path)
        self.config: SB3PipelineConfig = SB3PipelineConfig.from_yaml(
            "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
        )
        self.config.save_config.models_dir = str(self.temp_dir / "models")
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

    def test_pipeline_train_orchestrates_lifecycle(self) -> None:
        pipeline = SB3Pipeline(config=self.config, verbose=False)

        mock_env = mock.create_autospec(VecEnv, instance=True, spec_set=True)
        mock_model = mock.create_autospec(
            BaseAlgorithm, instance=True, spec_set=True
        )

        self.enter_context(
            mock.patch.object(
                pipeline.env_loader,
                "vec_env",
                return_value=mock_env,
                autospec=True,
                spec_set=True,
            )
        )
        self.enter_context(
            mock.patch.object(
                pipeline.model_loader,
                "model",
                return_value=mock_model,
                autospec=True,
                spec_set=True,
            )
        )

        trained_model = pipeline.train()

        self.assertIs(trained_model, mock_model)
        mock_model.learn.assert_called_once()
        mock_env.close.assert_called_once()

    @parameterized.parameters((None,), (999,))
    def test_init_callback_passes_eval_seed(
        self, eval_seed: int | None
    ) -> None:
        if self.config.vec_config is None:
            raise ValueError("vec_config must be present.")
        self.config.vec_config.seed = 42
        self.config.vec_config.eval_seed = eval_seed
        pipeline = SB3Pipeline(config=self.config, verbose=False)
        mock_env = mock.create_autospec(VecEnv, instance=True, spec_set=True)
        mock_eval_vec_env = self.enter_context(
            mock.patch.object(
                pipeline.env_loader,
                "eval_vec_env",
                return_value=mock_env,
                autospec=True,
                spec_set=True,
            )
        )
        self.enter_context(
            mock.patch.object(
                sb3_pipeline,
                "init_callback",
                return_value=[],
                autospec=True,
                spec_set=True,
            )
        )

        pipeline._init_callback()

        self.assertEqual(mock_eval_vec_env.call_args.kwargs["seed"], eval_seed)

    def test_evaluate_vec_env_uses_eval_seed_override(self) -> None:
        if self.config.vec_config is None:
            raise ValueError("vec_config must be present.")
        self.config.vec_config.seed = 42
        self.config.vec_config.eval_seed = None
        pipeline = SB3Pipeline(config=self.config, verbose=False)
        mock_env = mock.create_autospec(VecEnv, instance=True, spec_set=True)
        mock_model = mock.create_autospec(
            BaseAlgorithm, instance=True, spec_set=True
        )
        mock_eval_vec_env = self.enter_context(
            mock.patch.object(
                pipeline.env_loader,
                "eval_vec_env",
                return_value=mock_env,
                autospec=True,
                spec_set=True,
            )
        )
        self.enter_context(
            mock.patch.object(
                sb3_pipeline,
                "evaluate_policy",
                return_value=([1.0], [10]),
                autospec=True,
                spec_set=True,
            )
        )

        pipeline.evaluate(
            n_eval_episodes=1,
            save_to_file=False,
            checkpoint=mock_model,
            env="vec",
        )

        mock_eval_vec_env.assert_called_once_with(monitor_dir=None, seed=None)

    def test_env_loader_eval_vec_env_isolates_train_seed(self) -> None:
        if self.config.vec_config is None:
            raise ValueError("vec_config must be present.")
        self.config.vec_config.seed = 42
        pipeline = SB3Pipeline(config=self.config, verbose=False)
        mock_env = mock.create_autospec(VecEnv, instance=True, spec_set=True)
        mock_make = self.enter_context(
            mock.patch.object(
                sb3_loader,
                "make_vec_env",
                return_value=mock_env,
                autospec=True,
                spec_set=True,
            )
        )

        pipeline.env_loader.vec_env()
        pipeline.env_loader.eval_vec_env(seed=None)
        pipeline.env_loader.eval_vec_env(seed=123)

        seeds = [call.kwargs["seed"] for call in mock_make.call_args_list]
        self.assertEqual(seeds, [42, None, 123])
        self.assertEqual(pipeline.env_loader.vec_config.seed, 42)


if __name__ == "__main__":
    absltest.main()
