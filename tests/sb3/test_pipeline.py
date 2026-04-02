import os

import gymnasium as gym
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader
from rl_pipeline.sb3.callback import CheckpointCallbackConfig, EvalCallbackConfig
from rl_pipeline.sb3.config import (
    ArbitraryCallbackConfig,
    SB3CallbackConfig,
    SB3PipelineConfig,
)
from rl_pipeline.sb3.pipeline import init_callback


class _TestArbitraryCallback(BaseCallback):
    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag

    def _on_step(self) -> bool:
        return True


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
    # assert os.path.exists(pipeline.config.save_config.model_save_path)


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


def test_init_callback_appends_arbitrary_callbacks_in_order(tmp_path):
    callback_config = SB3CallbackConfig(
        eval_callback_config=EvalCallbackConfig(eval_freq=10, n_eval_episodes=2),
        ckpt_callback_config=CheckpointCallbackConfig(
            save_freq=10,
            save_path=str(tmp_path),
        ),
        arbitrary_callback_configs=[
            ArbitraryCallbackConfig(
                callback_class=_TestArbitraryCallback,
                callback_kwargs={"tag": "alpha"},
            )
        ],
    )

    eval_env = DummyVecEnv([lambda: gym.make("CartPole-v1", render_mode="rgb_array")])
    video_env = gym.make("CartPole-v1", render_mode="rgb_array")

    callbacks = init_callback(
        eval_env=eval_env,
        video_env=video_env,
        callback_config=callback_config,
    )

    eval_env.close()
    video_env.close()

    assert len(callbacks) == 3
    assert isinstance(callbacks[0], CheckpointCallback)
    assert callbacks[1].__class__.__name__ == "SuccessEvalCallback"
    assert isinstance(callbacks[2], _TestArbitraryCallback)
    assert callbacks[2].tag == "alpha"
