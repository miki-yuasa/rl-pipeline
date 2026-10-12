from __future__ import annotations

from unittest import mock

from absl.testing import absltest
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.sb3 import (
    SB3AlgorithmConfig,
    SB3ReplicatePipelineConfig,
    resolve_algorithm,
)


class _DummyAlgo(BaseAlgorithm):
    """Dummy algorithm class for resolution testing."""


def _mock_get_class(path: str) -> type[BaseAlgorithm]:
    if path == "stable_baselines3.RecurrentPPO":
        raise AttributeError("missing in sb3")
    if path == "sb3_contrib.RecurrentPPO":
        return _DummyAlgo
    raise ValueError(f"Unexpected class path: {path}")


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
        self.assertEqual(
            paths,
            ["out/rep_0/model.zip", "out/rep_1/model.zip"],
        )


if __name__ == "__main__":
    absltest.main()
