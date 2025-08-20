import os
from typing import Any

from pydantic import BaseModel, ConfigDict
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_pipeline.core.config import ConfigReader, SaveConfig, SaveConfigReader
from rl_pipeline.core.utils.io import format_large_number, read_config_dict_from_yaml
from rl_pipeline.gymnasium.config import MakeEnvConfig, WrapperConfig

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
    wrapper_config: WrapperConfig | None = None
    vec_config: MakeVecEnvConfig | None = None
    algo_config: SB3AlgorithmConfig
    learn_config: SB3LearnConfig
    callback_configs: SB3CallbackConfigs


class SB3ConfigReader(BaseModel, ConfigReader[SB3PipelineConfig]):
    """Configuration reader for SB3 pipeline."""

    device: str | int = "cuda:0"
    experiment_id: str = ""
    retrain_model: bool = True
    save_config: SaveConfigReader
    config_dir: str = "configs"
    env_config_file: str = "env_config.yaml"
    wrapper_config_file: str | None = None
    vec_config_file: str | None = None
    algo_config_file: str = "algo_config.yaml"
    learn_config_file: str = "learn_config.yaml"
    callback_configs_file: str = "callback_configs.yaml"

    def to_config(self) -> SB3PipelineConfig:
        device: str = (
            self.device if isinstance(self.device, str) else f"cuda:{self.device}"
        )

        env_config: MakeEnvConfig = read_config_dict_from_yaml(
            self.config_dir, self.env_config_file, MakeEnvConfig
        )
        wrapper_config: WrapperConfig | None = (
            read_config_dict_from_yaml(
                self.config_dir, self.wrapper_config_file, WrapperConfig
            )
            if self.wrapper_config_file
            else None
        )
        vec_config: MakeVecEnvConfig | None = (
            read_config_dict_from_yaml(
                self.config_dir, self.vec_config_file, MakeVecEnvConfig
            )
            if self.vec_config_file
            else None
        )
        algo_config: SB3AlgorithmConfig = read_config_dict_from_yaml(
            self.config_dir, self.algo_config_file, SB3AlgorithmConfig
        )
        learn_config: SB3LearnConfig = read_config_dict_from_yaml(
            self.config_dir, self.learn_config_file, SB3LearnConfig
        )
        callback_configs: SB3CallbackConfigs = read_config_dict_from_yaml(
            self.config_dir, self.callback_configs_file, SB3CallbackConfigs
        )

        save_config: SaveConfig = self.save_config.to_config(
            experiment_id=self.experiment_id,
            model_name_suffix="_" + format_large_number(learn_config.total_timesteps),
        )

        pipeline_config = SB3PipelineConfig(
            device=device,
            experiment_id=self.experiment_id,
            retrain_model=self.retrain_model,
            save_config=save_config,
            env_config=env_config,
            wrapper_config=wrapper_config,
            vec_config=vec_config,
            algo_config=algo_config,
            learn_config=learn_config,
            callback_configs=callback_configs,
        )

        return pipeline_config
