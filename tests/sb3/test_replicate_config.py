from __future__ import annotations

from unittest import mock

from absl.testing import absltest, parameterized

from rl_pipeline.sb3 import (
    MakeVecEnvConfig,
    SB3AlgorithmConfig,
    SB3PipelineConfig,
    SB3ReplicatePipelineConfig,
    replicate_helpers,
)


def _algo(cfg: SB3PipelineConfig) -> SB3AlgorithmConfig:
    if cfg.algo_config is None:
        raise ValueError("Unexpected None algo_config")
    return cfg.algo_config


def _vec(cfg: SB3PipelineConfig) -> MakeVecEnvConfig:
    if cfg.vec_config is None:
        raise ValueError("Unexpected None vec_config")
    return cfg.vec_config


class SB3ReplicateSeedConfigTest(parameterized.TestCase):
    def test_replicate_unseeded_samples_and_records_base_seed(self) -> None:
        self.enter_context(
            mock.patch.object(
                replicate_helpers,
                "sample_base_seed",
                side_effect=[111, 222],
                autospec=True,
                spec_set=True,
            )
        )
        raw_by_section = {
            "replicate_config": {"num_replicates": 2},
            "single_pipeline_config": {
                "algo_config": {"algorithm": "stable_baselines3.PPO"},
                "vec_config": {"n_envs": 4},
            },
        }
        rep_a = SB3ReplicatePipelineConfig.model_validate(raw_by_section)
        rep_b = SB3ReplicatePipelineConfig.model_validate(raw_by_section)
        self.assertEqual(rep_a.replicate_config.base_seed, 111)
        self.assertEqual(rep_b.replicate_config.base_seed, 222)

    @parameterized.parameters(("base_seed",), ("algo_seed",))
    def test_replicate_array_job_seed_matches_full_sweep(
        self, source: str
    ) -> None:
        base_seed = 17
        algo_cfg: dict[str, object] = {"algorithm": "stable_baselines3.PPO"}
        rep_cfg_by_key: dict[str, object] = {
            "num_replicates": 4,
            "replicate_start_id": 0,
        }
        if source == "base_seed":
            rep_cfg_by_key["base_seed"] = base_seed
        else:
            algo_cfg["seed"] = base_seed

        full = SB3ReplicatePipelineConfig.model_validate(
            {
                "replicate_config": rep_cfg_by_key,
                "single_pipeline_config": {
                    "algo_config": algo_cfg,
                    "vec_config": {"n_envs": 200},
                },
            }
        )
        array = SB3ReplicatePipelineConfig.model_validate(
            {
                "replicate_config": {
                    **rep_cfg_by_key,
                    "num_replicates": 1,
                    "replicate_start_id": 2,
                },
                "single_pipeline_config": {
                    "algo_config": algo_cfg,
                    "vec_config": {"n_envs": 200},
                },
            }
        )
        expected = full.ind_pipeline_configs[2]
        actual = array.ind_pipeline_configs[0]
        self.assertEqual(
            _algo(actual).algo_kwargs["seed"],
            _algo(expected).algo_kwargs["seed"],
        )
        self.assertEqual(_vec(actual).seed, _vec(expected).seed)

    def test_replicate_single_config_preserves_unseeded_state(self) -> None:
        rep_cfg = SB3ReplicatePipelineConfig.model_validate(
            {
                "replicate_config": {"num_replicates": 2},
                "single_pipeline_config": {
                    "experiment_id": "exp_{replicate_signature}",
                    "algo_config": {"algorithm": "stable_baselines3.PPO"},
                    "vec_config": {"n_envs": 2},
                },
            }
        )
        if rep_cfg.single_config is None:
            raise ValueError("Unexpected None single_config")
        self.assertIsNone(_algo(rep_cfg.single_config).algo_kwargs.get("seed"))
        self.assertIsNone(_vec(rep_cfg.single_config).seed)

    def test_replicate_rejects_negative_algo_seed(self) -> None:
        with self.assertRaises(ValueError):
            SB3ReplicatePipelineConfig.model_validate(
                {
                    "replicate_config": {"num_replicates": 2},
                    "single_pipeline_config": {
                        "algo_config": {
                            "algorithm": "stable_baselines3.PPO",
                            "seed": -5,
                        },
                    },
                }
            )


if __name__ == "__main__":
    absltest.main()
