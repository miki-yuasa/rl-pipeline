from pathlib import Path

import optuna
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_pipeline.experiment.optuna import build_dashboard_command
from rl_pipeline.sb3 import (
    SB3OptunaConfig,
    SB3OptunaDashboardConfig,
    SB3Pipeline,
    SB3PipelineConfigReader,
)
from rl_pipeline.sb3.config import SB3OptunaParamConfig
from rl_pipeline.sb3.experiment.optuna import (
    filter_algorithm_kwargs,
    sample_params_from_config,
)


def test_filter_algorithm_kwargs_drops_unknown_values():
    from stable_baselines3 import PPO

    kwargs = {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "unknown_param": 123,
    }

    filtered = filter_algorithm_kwargs(PPO, kwargs)

    assert "learning_rate" in filtered
    assert "gamma" in filtered
    assert "unknown_param" not in filtered


def test_dashboard_command_builder():
    command = build_dashboard_command(
        storage_url="sqlite:///study.db",
        host="0.0.0.0",
        port=8080,
    )

    assert command.command == [
        "optuna-dashboard",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "sqlite:///study.db",
    ]


def test_sample_params_from_config_supports_nested_and_mapping():
    trial = optuna.trial.FixedTrial(
        {
            "exp_n_steps": 5,
            "gamma_eps": 0.01,
            "activation": "relu",
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
    ]

    params = sample_params_from_config(trial, tune_params)

    assert params["n_steps"] == 32
    assert params["gamma"] == 0.99
    assert params["policy_kwargs"]["activation_fn"] is nn.ReLU


def test_sb3_pipeline_optimize_smoke(tmp_path: Path):
    config = SB3PipelineConfigReader.from_yaml(
        "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
    ).to_config()

    assert config.vec_config is not None
    config.vec_config.vec_env_cls = DummyVecEnv
    config.vec_config.n_envs = 1

    db_path = tmp_path / "study.db"
    config.optuna_config = SB3OptunaConfig(
        storage_url=f"sqlite:///{db_path}",
        study_name="sb3_smoke_optuna",
        n_trials=1,
        n_jobs=1,
        n_startup_trials=1,
        n_warmup_steps=0,
        n_evaluations=1,
        n_eval_episodes=1,
        total_timesteps=64,
        tune_params=[
            SB3OptunaParamConfig(
                name="learning_rate",
                suggest_type="float",
                low=1e-5,
                high=1e-2,
                log=True,
            )
        ],
        dashboard=SB3OptunaDashboardConfig(launch=False),
    )

    pipeline = SB3Pipeline(config=config, verbose=False)
    study = pipeline.optimize()

    assert len(study.trials) == 1
    assert study.study_name == "sb3_smoke_optuna"
    assert db_path.exists()
