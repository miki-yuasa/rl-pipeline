from __future__ import annotations

from typing import TypeVar
from unittest import mock

from absl.testing import absltest
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.gymnasium.config import (
    BaseWrapperConfig,
    WrapperConfig,
    load_wrapper_config,
)
from rl_pipeline.sb3 import (
    SB3AlgorithmConfig,
    SB3PipelineConfig,
    SB3ReplicatePipelineConfig,
    resolve_algorithm,
)

_T = TypeVar("_T")


class _DummyAlgo(BaseAlgorithm):
    """Dummy algorithm class for resolution testing."""


class _DummyWrapper(Wrapper):
    """Gymnasium wrapper dummy for testing."""


class _DummyWrapperConfig(BaseWrapperConfig):
    """Configuration model for dummy wrapper tests."""

    scale: float = 1.0

    def to_wrapper_config(self) -> WrapperConfig:
        """Standard wrapper configuration conversion."""
        return WrapperConfig(
            wrapper_class=_DummyWrapper,
            wrapper_kwargs={"scale": self.scale},
        )


class _CustomPipelineConfig(SB3PipelineConfig):
    """Custom SB3 pipeline configuration subclass for replication testing."""

    custom_flag: bool = True


def _mock_get_class(path: str) -> type[BaseAlgorithm]:
    if path == "stable_baselines3.RecurrentPPO":
        raise AttributeError("missing in sb3")
    if path == "sb3_contrib.RecurrentPPO":
        return _DummyAlgo
    raise ValueError(f"Unexpected class path: {path}")


def _require(val: _T | None) -> _T:
    if val is None:
        raise ValueError("Unexpected None value")
    return val


class SB3ConfigTest(absltest.TestCase):
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
        seeds = [
            cfg.algo_config.algo_kwargs["seed"]
            for cfg in configs
            if cfg.algo_config is not None
        ]
        self.assertEqual(
            paths,
            ["out/rep_0/model.zip", "out/rep_1/model.zip"],
        )
        self.assertEqual(seeds, [10, 20])

    def test_replicate_paths_resolution(self) -> None:
        cfg_by_key = {
            "replicate_config": {
                "num_replicates": 2,
                "replicate_start_id": 0,
                "replicate_signature": "rep_{rep_id}",
            },
            "single_pipeline_config": {
                "experiment_id": "test_rep_exp",
                "save_config": {"models_dir": "out", "model_name": "agent"},
                "algo_config": {"algorithm": "stable_baselines3.PPO"},
                "vec_config": {"n_envs": 2},
                "callback_config": {
                    "eval_callback_config": {
                        "eval_freq": 100,
                        "n_eval_episodes": 10,
                        "log_path": "eval",
                    },
                    "ckpt_callback_config": {
                        "save_freq": 1000,
                        "save_path": "ckpts",
                    },
                },
            },
        }
        rep_cfg = SB3ReplicatePipelineConfig.model_validate(cfg_by_key)
        rep0 = rep_cfg.ind_pipeline_configs[0]
        rep1 = rep_cfg.ind_pipeline_configs[1]
        rep0_vec = _require(rep0.vec_config)
        rep0_cb = _require(rep0.callback_config)
        rep1_cb = _require(rep1.callback_config)
        rep1_ckpt = _require(rep1_cb.ckpt_callback_config)
        self.assertEqual(
            rep0_vec.monitor_dir,
            "out/test_rep_exp/agent_1.0M/rep_0/monitor",
        )
        self.assertEqual(
            rep0_cb.eval_callback_config.log_path,
            "out/test_rep_exp/agent_1.0M/rep_0/eval",
        )
        self.assertEqual(
            rep1_ckpt.save_path,
            "out/test_rep_exp/agent_1.0M/rep_1/ckpts",
        )

    def test_load_wrapper_config_with_base_wrapper_config(
        self,
    ) -> None:
        res = _require(
            load_wrapper_config({"_target_": _DummyWrapperConfig, "scale": 2.5})
        )
        self.assertIs(res.wrapper_class, _DummyWrapper)
        self.assertEqual(res.wrapper_kwargs, {"scale": 2.5})

    def test_replicate_with_custom_pipeline_target(self) -> None:
        cfg_by_key = {
            "replicate_config": {
                "num_replicates": 2,
                "replicate_start_id": 0,
                "replicate_signature": "rep_{rep_id}",
            },
            "single_pipeline_config": {
                "_target_": _CustomPipelineConfig,
                "experiment_id": "custom_{replicate_signature}",
                "algo_config": {"algorithm": "stable_baselines3.PPO"},
            },
        }
        rep_cfg = SB3ReplicatePipelineConfig.model_validate(cfg_by_key)
        self.assertLen(rep_cfg.ind_pipeline_configs, 2)
        self.assertIsInstance(
            rep_cfg.ind_pipeline_configs[0], _CustomPipelineConfig
        )


if __name__ == "__main__":
    absltest.main()
