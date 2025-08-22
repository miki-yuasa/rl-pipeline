import pytest

from rl_pipeline.sb3 import SB3PipelineConfig, SB3PipelineConfigReader


def test_sb3_pipeline_config_reader():
    config_file_path: str = "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"

    pipeline_config = SB3PipelineConfigReader.from_yaml(config_file_path).to_config()

    assert isinstance(pipeline_config, SB3PipelineConfig)
