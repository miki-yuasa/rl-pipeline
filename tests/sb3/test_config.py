from __future__ import annotations

from typing import Any
from unittest import mock

from absl.testing import absltest
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.sb3 import (
    ArbitraryCallbackConfig,
    SB3AlgorithmConfig,
    SB3OptunaConfig,
    SB3OptunaParamConfig,
    SB3PipelineConfig,
    SB3ReplicatePipelineConfig,
    resolve_algorithm,
)


class _DummyAlgo(BaseAlgorithm):
    """Dummy algorithm class for resolution testing."""


def _dummy_sample_params(_trial: Any) -> dict[str, float]:
    return {"learning_rate": 1e-3}


def _mock_get_class(path: str) -> type[BaseAlgorithm]:
    if path == "stable_baselines3.RecurrentPPO":
        raise AttributeError("missing in sb3")
    if path == "sb3_contrib.RecurrentPPO":
        return _DummyAlgo
    raise AssertionError(f"Unexpected class path: {path}")


class SB3ConfigTest(absltest.TestCase):
    def test_pipeline_config_from_yaml_metadata(self) -> None:
        config = SB3PipelineConfig.from_yaml(
            "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
        )
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.experiment_id, "1.a")
        self.assertFalse(config.retrain_model)

    def test_pipeline_config_from_yaml_algorithm_and_paths(self) -> None:
        config = SB3PipelineConfig.from_yaml(
            "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
        )
        assert config.algo_config is not None
        self.assertIs(config.algo_config.algorithm, PPO)
        self.assertIn("cartpole_model", config.save_config.model_save_path)

    def test_resolve_algorithm_falls_back_to_sb3_contrib(self) -> None:
        from rl_pipeline.sb3 import algo_config

        self.enter_context(
            mock.patch.object(
                algo_config,
                "get_class",
                side_effect=_mock_get_class,
                autospec=True,
                spec_set=True,
            )
        )
        algo_cls = resolve_algorithm("RecurrentPPO")
        self.assertIs(algo_cls, _DummyAlgo)

    def test_resolve_algorithm_raises_when_not_found(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not resolve"):
            resolve_algorithm("MissingAlgoNonExistent")

    def test_sb3_algorithm_config_hydra_target_pattern(self) -> None:
        cfg = SB3AlgorithmConfig(
            _target_="stable_baselines3.PPO",
            learning_rate=0.0003,
        )
        self.assertIs(cfg.algorithm, PPO)
        self.assertEqual(cfg.algo_kwargs["learning_rate"], 0.0003)

    def test_arbitrary_callback_config(self) -> None:
        config = ArbitraryCallbackConfig(
            callback_class="stable_baselines3.common.callbacks.CheckpointCallback",
            callback_kwargs={"save_freq": 10, "save_path": "tmp"},
        )
        assert isinstance(config.callback_class, type)
        self.assertEqual(config.callback_class.__name__, "CheckpointCallback")
        self.assertEqual(
            config.callback_kwargs, {"save_freq": 10, "save_path": "tmp"}
        )

    def test_optuna_config_resolves_sample_function(self) -> None:
        config = SB3OptunaConfig(
            storage_url="sqlite:///tmp.db",
            n_trials=2,
            sample_params_fn="tests.sb3.test_config._dummy_sample_params",
        )
        self.assertEqual(config.storage_url, "sqlite:///tmp.db")
        assert callable(config.sample_params_fn)
        self.assertEqual(config.sample_params_fn(None), {"learning_rate": 1e-3})

    def test_optuna_config_with_tune_params(self) -> None:
        config = SB3OptunaConfig(
            storage_url="sqlite:///tmp.db",
            tune_params=[
                SB3OptunaParamConfig(
                    name="gamma_eps",
                    suggest_type="float",
                    low=0.0001,
                    high=0.1,
                    log=True,
                    one_minus=True,
                    target="gamma",
                ),
                SB3OptunaParamConfig(
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
        self.assertLen(config.tune_params, 2)
        assert config.tune_params[1].value_mapping is not None
        self.assertEqual(
            config.tune_params[1].value_mapping["relu"].__name__, "ReLU"
        )

    def test_pipeline_config_pure_instantiation(self) -> None:
        config = SB3PipelineConfig(
            experiment_id="pure_test",
            algo_config=SB3AlgorithmConfig(algorithm=PPO),
        )
        self.assertEqual(config.experiment_id, "pure_test")
        self.assertNotIn("env_config_file", SB3PipelineConfig.model_fields)

    def test_replicate_templating_with_inlined_wrapper(self) -> None:
        cfg_by_key = {
            "replicate_config": {
                "num_replicates": 2,
                "replicate_start_id": 0,
                "replicate_signature": "rep_{rep_id}",
            },
            "single_pipeline_config": {
                "experiment_id": "rep_test",
                "algo_config": {"algorithm": "stable_baselines3.PPO"},
                "wrapper_config": {
                    "wrapper_class": "gymnasium.Wrapper",
                    "wrapper_kwargs": {
                        "checkpoint_path": "out/{replicate_signature}/model.zip"
                    },
                },
            },
        }
        rep_cfg = SB3ReplicatePipelineConfig.model_validate(cfg_by_key)
        configs = rep_cfg.ind_pipeline_configs
        paths = [
            cfg.wrapper_config.wrapper_kwargs["checkpoint_path"]
            for cfg in configs
            if cfg.wrapper_config is not None
        ]
        self.assertEqual(
            paths,
            ["out/rep_0/model.zip", "out/rep_1/model.zip"],
        )


if __name__ == "__main__":
    absltest.main()
