from typing import Any, Callable

from gymnasium import Env, Wrapper
from pydantic import BaseModel, ConfigDict
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_pipeline.core.config import SaveConfig
from rl_pipeline.gymnasium.config import MakeEnvConfig, WrapperConfig
from rl_pipeline.utils.io import get_class

from .callback import CheckpointCallbackConfig, EvalCallbackConfig


class MakeVecEnvConfig(BaseModel):
    n_envs: int = 1
    seed: int | None = None
    start_index: int = 0
    monitor_dir: str | None = None
    env_kwargs: dict[str, Any] | None = None
    vec_env_cls: type[SubprocVecEnv] | type[DummyVecEnv] | None = SubprocVecEnv
    vec_env_kwargs: dict[str, Any] | None = None
    monitor_kwargs: dict[str, Any] | None = None

    # allow arbitrary kwargs
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SB3AlgorithmConfig(BaseModel):
    algorithm: type[BaseAlgorithm]
    algo_kwargs: dict[str, Any]


class SB3LearnConfig(BaseModel):
    total_timesteps: int
    callback: MaybeCallback = None
    log_interval: int = 100
    tb_log_name: str = "run"
    reset_num_timesteps: bool = True
    progress_bar: bool = False

    # allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SB3CallbackConfigs(BaseModel):
    eval_callback_config: EvalCallbackConfig
    ckpt_callback_config: CheckpointCallbackConfig


class SB3PipelineConfig(BaseModel):
    device: str = "cuda:0"
    experiment_id: str = ""
    retrain_model: bool = True
    save_config: SaveConfig
    env_config: MakeEnvConfig
    wrapper_config: WrapperConfig
    vec_config: MakeVecEnvConfig
    algo_config: SB3AlgorithmConfig
    learn_config: SB3LearnConfig
