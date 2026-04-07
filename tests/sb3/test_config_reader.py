import json
from typing import Any

import pytest
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.sb3 import (
    SB3AlgorithmConfigReader,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
    config_reader,
)


def _dummy_sample_params(_trial):
    return {"learning_rate": 1e-3}


def test_sb3_pipeline_config_reader():
    target_read_config: dict[str, Any] = {
        "config_dir": "tests/sb3/assets/configs",
        "device": 0,
        "env_config_file": "cartpole_env_config.yaml",
        "experiment_id": "1.a",
        "model_config_file": "ppo.yaml",
        "optuna_config": None,
        "retrain_model": False,
        "save_config": {
            "animation_dir": "animations",
            "animation_ext": "gif",
            "best_model_filename": "best_model.zip",
            "eval_dir": "eval",
            "eval_metrics_filename": "eval_metrics.yaml",
            "include_suffix_in_filename": True,
            "model_filename": "final_model.zip",
            "model_name": "cartpole_model",
            "models_dir": "tests/sb3/out/models",
            "monitor_dir": "monitor",
            "tb_dir": "tb",
        },
        "wrapper_config_file": None,
        "experiment_manager_config": {
            "callback_config": {"gradient_save_freq": 50},
            "manager_class": "rl_pipeline.sb3.SB3WandbExperimentManager",
            "manager_config": {
                "name": "cartpole_ppo",
                "monitor_gym": False,
                "project": "rl_pipeline_test",
                "save_code": False,
                "sync_tensorboard": True,
                "entity": "miki-yuasa-university-of-illinois-urbana-champaign",
            },
        },
    }
    config_file_path: str = "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"

    config_reader = SB3PipelineConfigReader.from_yaml(config_file_path)

    assert config_reader.model_dump() == target_read_config


def test_sb3_pipeline_config():
    target_json: str = "tests/sb3/assets/configs/pipeline_target_config.json"
    config_file_path: str = "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"

    pipeline_config: SB3PipelineConfig = SB3PipelineConfigReader.from_yaml(
        config_file_path
    ).to_config()

    with open(target_json, "r") as f:
        target_config: dict[str, Any] = json.load(f)

    json_config: dict[str, Any] = json.loads(pipeline_config.model_dump_json())

    assert json_config == target_config


def test_sb3_algorithm_config_reader_falls_back_to_sb3_contrib(monkeypatch):
    reader = SB3AlgorithmConfigReader(algorithm="RecurrentPPO")

    class _DummyAlgo(BaseAlgorithm):
        pass

    def _mock_get_class(path: str):
        if path == "stable_baselines3.RecurrentPPO":
            raise AttributeError("missing in sb3")
        if path == "sb3_contrib.RecurrentPPO":
            return _DummyAlgo
        raise AssertionError(f"Unexpected class path: {path}")

    monkeypatch.setattr(config_reader, "get_class", _mock_get_class)
    monkeypatch.setattr(config_reader.importlib.util, "find_spec", lambda _: object())

    config = reader.to_config()

    assert config.algorithm is _DummyAlgo


def test_sb3_algorithm_config_reader_raises_when_not_found(monkeypatch):
    reader = SB3AlgorithmConfigReader(algorithm="MissingAlgo")

    def _mock_get_class(_path: str):
        raise AttributeError("missing")

    monkeypatch.setattr(config_reader, "get_class", _mock_get_class)
    monkeypatch.setattr(config_reader.importlib.util, "find_spec", lambda _: None)

    with pytest.raises(AssertionError, match="MissingAlgo"):
        reader.to_config()


def test_arbitrary_callback_config_reader_to_config():
    reader = config_reader.ArbitraryCallbackConfigReader(
        callback_class="stable_baselines3.common.callbacks.CheckpointCallback",
        callback_kwargs={"save_freq": 10, "save_path": "tmp"},
    )

    config = reader.to_config()

    assert config.callback_class.__name__ == "CheckpointCallback"
    assert config.callback_kwargs == {"save_freq": 10, "save_path": "tmp"}


def test_arbitrary_callback_config_reader_rejects_non_callback_class():
    reader = config_reader.ArbitraryCallbackConfigReader(
        callback_class="pathlib.Path",
    )

    with pytest.raises(AssertionError, match="must inherit from BaseCallback"):
        reader.to_config()


def test_optuna_config_reader_to_config_resolves_sample_function():
    reader = config_reader.SB3OptunaConfigReader(
        storage_url="sqlite:///tmp.db",
        n_trials=2,
        sample_params_fn="tests.sb3.test_config_reader._dummy_sample_params",
    )

    config = reader.to_config()

    assert config.storage_url == "sqlite:///tmp.db"
    assert config.n_trials == 2
    assert config.sample_params_fn is not None
    assert config.sample_params_fn.__name__ == _dummy_sample_params.__name__
    assert config.sample_params_fn(None) == _dummy_sample_params(None)


def test_optuna_config_reader_to_config_with_tune_params():
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

    assert len(config.tune_params) == 2
    assert config.tune_params[0].target == "gamma"
    assert config.tune_params[0].one_minus is True
    assert config.tune_params[1].value_mapping is not None
    assert config.tune_params[1].value_mapping["relu"].__name__ == "ReLU"
