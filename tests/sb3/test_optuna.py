from __future__ import annotations

import optuna
from absl.testing import absltest, parameterized
from torch import nn

from rl_pipeline.sb3 import SB3OptunaParamConfig
from rl_pipeline.sb3.experiment.optuna import sample_params_from_config


class SB3OptunaTest(parameterized.TestCase):
    def test_sample_params_from_config_transforms_and_indexes(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "exp_n_steps": 5,
                "gamma_eps": 0.01,
                "activation": "relu",
                "L_min": 10.0,
                "L_max": 80.0,
            }
        )
        tune_params = [
            SB3OptunaParamConfig(
                name="exp_n_steps",
                suggest_type="pow2_int",
                low=3,
                high=10,
                target="n_steps",
            ),
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
                value_mapping={"tanh": nn.Tanh, "relu": nn.ReLU},
                target="policy_kwargs.activation_fn",
            ),
            SB3OptunaParamConfig(
                name="L_min",
                suggest_type="float",
                low=1.0,
                high=20.0,
                target="wrapper.args.gain.range.0",
            ),
            SB3OptunaParamConfig(
                name="L_max",
                suggest_type="float",
                low=50.0,
                high=100.0,
                target="wrapper.args.gain.range.1",
            ),
        ]
        params = sample_params_from_config(trial, tune_params)
        self.assertEqual((params["n_steps"], params["gamma"]), (32, 0.99))
        self.assertIs(params["policy_kwargs"]["activation_fn"], nn.ReLU)
        self.assertEqual(params["wrapper"]["args"]["gain"]["range"], [10.0, 80.0])

    def test_sample_params_from_config_prunes_invalid_range(self) -> None:
        trial = optuna.trial.FixedTrial({"bad_min": 100.0, "bad_max": 10.0})
        tune_params = [
            SB3OptunaParamConfig(
                name="bad_min",
                target="wrapper.args.gain.range.0",
                suggest_type="float",
                low=100.0,
                high=100.0,
            ),
            SB3OptunaParamConfig(
                name="bad_max",
                target="wrapper.args.gain.range.1",
                suggest_type="float",
                low=10.0,
                high=10.0,
            ),
        ]
        with self.assertRaises(optuna.TrialPruned):
            sample_params_from_config(trial, tune_params)


if __name__ == "__main__":
    absltest.main()
