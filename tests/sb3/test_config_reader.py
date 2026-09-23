from __future__ import annotations

from typing import Any
from unittest import mock

from absl.testing import absltest
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.sb3 import (
    SB3AlgorithmConfigReader,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
    config_reader,
)


def _dummy_sample_params(_trial: Any) -> dict[str, float]:
    return {"learning_rate": 1e-3}


class SB3ConfigReaderTest(absltest.TestCase):
    def test_pipeline_config_reader_to_config(self) -> None:
        config_file_path = (
            "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
        )
        reader = SB3PipelineConfigReader.from_yaml(config_file_path)
        config: SB3PipelineConfig = reader.to_config()

        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.experiment_id, "1.a")
        self.assertFalse(config.retrain_model)
        self.assertIs(config.algo_config.algorithm, PPO)
        self.assertIn("tests/sb3/out/models", config.save_config.model_save_dir)
        self.assertIn("cartpole_model", config.save_config.model_save_path)

    def test_sb3_algorithm_config_reader_falls_back_to_sb3_contrib(
        self,
    ) -> None:
        reader = SB3AlgorithmConfigReader(algorithm="RecurrentPPO")

        class _DummyAlgo(BaseAlgorithm):
            pass

        def _mock_get_class(path: str) -> type[BaseAlgorithm]:
            if path == "stable_baselines3.RecurrentPPO":
                raise AttributeError("missing in sb3")
            if path == "sb3_contrib.RecurrentPPO":
                return _DummyAlgo
            raise AssertionError(f"Unexpected class path: {path}")

        self.enter_context(
            mock.patch.object(
                config_reader, "get_class", side_effect=_mock_get_class
            )
        )
        self.enter_context(
            mock.patch.object(
                config_reader.importlib.util, "find_spec", return_value=object()
            )
        )

        config = reader.to_config()
        self.assertIs(config.algorithm, _DummyAlgo)

    def test_sb3_algorithm_config_reader_raises_when_not_found(self) -> None:
        reader = SB3AlgorithmConfigReader(algorithm="MissingAlgo")

        self.enter_context(
            mock.patch.object(
                config_reader,
                "get_class",
                side_effect=AttributeError("missing"),
            )
        )
        self.enter_context(
            mock.patch.object(
                config_reader.importlib.util, "find_spec", return_value=None
            )
        )

        with self.assertRaisesRegex(AssertionError, "MissingAlgo"):
            reader.to_config()

    def test_arbitrary_callback_config_reader_to_config(self) -> None:
        reader = config_reader.ArbitraryCallbackConfigReader(
            callback_class="stable_baselines3.common.callbacks.CheckpointCallback",
            callback_kwargs={"save_freq": 10, "save_path": "tmp"},
        )
        config = reader.to_config()

        self.assertEqual(config.callback_class.__name__, "CheckpointCallback")
        self.assertEqual(
            config.callback_kwargs, {"save_freq": 10, "save_path": "tmp"}
        )

    def test_arbitrary_callback_config_reader_rejects_non_callback_class(
        self,
    ) -> None:
        reader = config_reader.ArbitraryCallbackConfigReader(
            callback_class="pathlib.Path",
        )
        with self.assertRaisesRegex(
            AssertionError, "must inherit from BaseCallback"
        ):
            reader.to_config()

    def test_optuna_config_reader_to_config_resolves_sample_function(
        self,
    ) -> None:
        reader = config_reader.SB3OptunaConfigReader(
            storage_url="sqlite:///tmp.db",
            n_trials=2,
            sample_params_fn="tests.sb3.test_config_reader._dummy_sample_params",
        )
        config = reader.to_config()

        self.assertEqual(config.storage_url, "sqlite:///tmp.db")
        self.assertEqual(config.n_trials, 2)
        self.assertEqual(config.parallel_backend, "process")
        self.assertIsNotNone(config.sample_params_fn)
        assert config.sample_params_fn is not None
        self.assertEqual(config.sample_params_fn(None), {"learning_rate": 1e-3})

    def test_optuna_config_reader_to_config_with_tune_params(self) -> None:
        reader = config_reader.SB3OptunaConfigReader(
            storage_url="sqlite:///tmp.db",
            tune_params=[
                config_reader.SB3OptunaParamConfigReader(
                    name="gamma_eps",
                    suggest_type="float",
                    low=0.0001,
                    high=0.1,
                    log=True,
                    one_minus=True,
                    target="gamma",
                ),
                config_reader.SB3OptunaParamConfigReader(
                    name="activation",
                    suggest_type="categorical",
                    choices=["tanh", "relu"],
                    value_mapping={
                        "tanh": "torch.nn.Tanh",
                        "relu": "torch.nn.ReLU",
                    },
                    target="policy_kwargs.activation_fn",
                ),
            ],
        )
        config = reader.to_config()

        self.assertLen(config.tune_params, 2)
        self.assertEqual(config.tune_params[0].target, "gamma")
        self.assertTrue(config.tune_params[0].one_minus)
        self.assertIsNotNone(config.tune_params[1].value_mapping)
        assert config.tune_params[1].value_mapping is not None
        self.assertEqual(
            config.tune_params[1].value_mapping["relu"].__name__, "ReLU"
        )


if __name__ == "__main__":
    absltest.main()
