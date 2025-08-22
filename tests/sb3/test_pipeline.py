import os

import pytest
import torch

from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader
from rl_pipeline.sb3.config import SB3PipelineConfig


@pytest.fixture
def sb3_pipeline_config():
    config_file_path = "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
    return SB3PipelineConfigReader.from_yaml(config_file_path).to_config()


def test_sb3_pipeline_initialization(sb3_pipeline_config: SB3PipelineConfig):
    pipeline = SB3Pipeline(config=sb3_pipeline_config)
    assert pipeline.config == sb3_pipeline_config


def test_sb3_pipeline_train(sb3_pipeline_config: SB3PipelineConfig):
    pipeline = SB3Pipeline(config=sb3_pipeline_config)
    pipeline.train()
    assert os.path.exists(pipeline.config.save_config.model_save_path)


def test_sb3_pipeline_load_model(sb3_pipeline_config: SB3PipelineConfig):
    pipeline = SB3Pipeline(config=sb3_pipeline_config)
    model = pipeline.load_model("best")
    assert model.device == torch.device(pipeline.config.device)


def test_sb3_pipeline_evaluate(sb3_pipeline_config: SB3PipelineConfig):
    pipeline = SB3Pipeline(config=sb3_pipeline_config)
    eval_results = pipeline.evaluate(checkpoint="best")
    assert eval_results is not None


def test_sb3_pipeline_record_replay(sb3_pipeline_config: SB3PipelineConfig):
    pipeline = SB3Pipeline(config=sb3_pipeline_config)
    model = pipeline.load_model("best")
    pipeline.record_replay(model)
    assert os.path.exists(pipeline.config.save_config.animation_save_path)
