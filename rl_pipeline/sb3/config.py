from typing import Any, Generic

from pydantic import BaseModel, ConfigDict, Field, field_serializer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_pipeline.core import SaveConfig
from rl_pipeline.gymnasium.config import MakeEnvConfig, WrapperConfig

from .callback import (
    CheckpointCallbackConfig,
    EvalCallbackConfig,
    VideoRecorderCallbackConfig,
)
from .experiment import SB3ExperimentManagerConfig


def class_to_string(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


class MakeVecEnvConfig(BaseModel):
    n_envs: int = Field(ge=1, default=1)
    seed: int | None = None
    start_index: int = Field(ge=0, default=0)
    monitor_dir: str | None = None
    vec_env_cls: type[SubprocVecEnv] | type[DummyVecEnv] | None = SubprocVecEnv
    vec_env_kwargs: dict[str, Any] | None = None
    monitor_kwargs: dict[str, Any] | None = None

    # allow arbitrary kwargs
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("vec_env_cls", when_used="json")
    def serialize_vec_env_cls(
        self, vec_env_cls: type[SubprocVecEnv | DummyVecEnv]
    ) -> str:
        return class_to_string(vec_env_cls)


class SB3AlgorithmConfig(BaseModel):
    algorithm: type[BaseAlgorithm]
    algo_kwargs: dict[str, Any]

    # For json serialization, `algorithm` should be in a string (e.b. "stable_baselines.PPO")
    @field_serializer("algorithm", when_used="json")
    def serialize_algorithm(self, algorithm: type[BaseAlgorithm]) -> str:
        return class_to_string(algorithm)


class SB3LearnConfig(BaseModel):
    total_timesteps: int = Field(ge=1, default=1_000_000)
    # callback: MaybeCallback = None
    log_interval: int = Field(ge=0, default=1)
    tb_log_name: str = "run"
    reset_num_timesteps: bool = True
    progress_bar: bool = False

    # allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SB3CallbackConfig(BaseModel):
    eval_callback_config: EvalCallbackConfig
    ckpt_callback_config: CheckpointCallbackConfig
    video_recorder_callback_config: VideoRecorderCallbackConfig | None = None


class SB3PipelineConfig(BaseModel):
    device: str = "cuda:0"
    experiment_id: str = ""
    retrain_model: bool = True
    save_config: SaveConfig
    env_config: MakeEnvConfig
    wrapper_config: WrapperConfig | None = None
    vec_config: MakeVecEnvConfig | None = None
    algo_config: SB3AlgorithmConfig
    learn_config: SB3LearnConfig
    callback_config: SB3CallbackConfig
    experiment_manager_config: SB3ExperimentManagerConfig | None = None


class SB3ModelConfig(BaseModel):
    algo_config: SB3AlgorithmConfig
    learn_config: SB3LearnConfig
    vec_config: MakeVecEnvConfig | None = None
    callback_config: SB3CallbackConfig
