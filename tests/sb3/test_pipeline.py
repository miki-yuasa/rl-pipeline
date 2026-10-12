from __future__ import annotations

from pathlib import Path
from unittest import mock

from absl.testing import absltest
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfig


class SB3PipelineTest(absltest.TestCase):
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


if __name__ == "__main__":
    absltest.main()
