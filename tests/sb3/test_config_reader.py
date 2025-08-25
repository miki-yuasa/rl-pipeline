import json
from typing import Any

import pytest

from rl_pipeline.sb3 import SB3PipelineConfig, SB3PipelineConfigReader


def test_sb3_pipeline_config_reader():
    target_read_config: dict[str, Any] = {
        "config_dir": "tests/sb3/assets/configs",
        "device": 0,
        "env_config_file": "cartpole_env_config.yaml",
        "experiment_id": "1.a",
        "model_config_file": "ppo.yaml",
        "retrain_model": False,
        "save_config": {
            "animation_dir": "animations",
            "animation_ext": "gif",
            "best_model_filename": "best_model.zip",
            "eval_dir": "eval",
            "eval_metrics_filename": "eval_metrics.yaml",
            "include_model_name_suffix": True,
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
