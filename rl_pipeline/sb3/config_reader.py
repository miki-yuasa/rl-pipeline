import os
from typing import Any

from pydantic import BaseModel, Field
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_pipeline.core import (
    ConfigReader,
    ReplicateConfig,
    SaveConfig,
    SaveConfigReader,
    YAMLReaderMixin,
)
from rl_pipeline.core.utils.io import (
    format_large_number,
    get_class,
    read_config_dict_from_yaml,
)
from rl_pipeline.gymnasium.config import (
    MakeEnvConfig,
    WrapperConfig,
    WrapperConfigReader,
)

from .callback import (
    CheckpointCallbackConfig,
    EvalCallbackConfig,
    VideoRecorderCallbackConfig,
)
from .config import (
    MakeVecEnvConfig,
    SB3AlgorithmConfig,
    SB3CallbackConfig,
    SB3ExperimentManagerConfig,
    SB3LearnConfig,
    SB3ModelConfig,
    SB3PipelineConfig,
    SB3ReplicatePipelineConfig,
)
from .experiment.base import SB3ExperimentManager


class SB3AlgorithmConfigReader(
    BaseModel, ConfigReader[SB3AlgorithmConfig], YAMLReaderMixin
):
    algorithm: str = "PPO"
    algo_kwargs: dict[str, Any] = {}

    def to_config(self) -> SB3AlgorithmConfig:
        algo_class: type[BaseAlgorithm] | None = get_class(
            "stable_baselines3." + self.algorithm
        )
        assert algo_class is not None, (
            f"Could not find algorithm class for {self.algorithm}"
        )
        return SB3AlgorithmConfig(
            algorithm=algo_class,
            algo_kwargs=self.algo_kwargs,
        )


class SB3LearnConfigReader(BaseModel, ConfigReader[SB3LearnConfig], YAMLReaderMixin):
    total_timesteps: int = Field(ge=1, default=1_000_000)
    log_interval: int = Field(ge=0, default=100)
    tb_log_name: str = "run"
    reset_num_timesteps: bool = True
    progress_bar: bool = False

    def to_config(self) -> SB3LearnConfig:
        return SB3LearnConfig(
            total_timesteps=self.total_timesteps,
            log_interval=self.log_interval,
            tb_log_name=self.tb_log_name,
            reset_num_timesteps=self.reset_num_timesteps,
            progress_bar=self.progress_bar,
        )


class MakeVecEnvConfigReader(BaseModel, YAMLReaderMixin):
    n_envs: int = Field(ge=1, default=1)
    seed: int | None = None
    start_index: int = Field(ge=0, default=0)
    vec_env_cls: str | None = "SubprocVecEnv"
    vec_env_kwargs: dict[str, Any] | None = None
    monitor_kwargs: dict[str, Any] | None = None

    def to_config(self, save_config: SaveConfig) -> MakeVecEnvConfig:
        vec_env_cls: type[SubprocVecEnv] | type[DummyVecEnv] | None = (
            get_class("stable_baselines3.common.vec_env." + self.vec_env_cls)
            if self.vec_env_cls
            else None
        )
        return MakeVecEnvConfig(
            n_envs=self.n_envs,
            seed=self.seed,
            start_index=self.start_index,
            monitor_dir=save_config.monitor_save_dir,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=self.vec_env_kwargs,
            monitor_kwargs=self.monitor_kwargs,
        )


class EvalCallbackConfigReader(BaseModel, YAMLReaderMixin):
    eval_freq: int = Field(ge=0)
    n_eval_episodes: int = Field(ge=1)
    log_path: str = "eval"
    deterministic: bool = False
    render: bool = False

    def to_config(self, save_config: SaveConfig) -> EvalCallbackConfig:
        return EvalCallbackConfig(
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            best_model_save_path=save_config.model_save_dir,
            log_path=os.path.join(save_config.model_save_dir, self.log_path),
            deterministic=self.deterministic,
            render=self.render,
        )


class CheckpointCallbackConfigReader(BaseModel, YAMLReaderMixin):
    save_freq: int = Field(ge=1, default=100)
    save_path: str = "ckpts"
    name_prefix: str = "ckpt"
    save_replay_buffer: bool = False
    verbose: int = 0

    def to_config(self, save_config: SaveConfig) -> CheckpointCallbackConfig:
        return CheckpointCallbackConfig(
            save_freq=self.save_freq,
            save_path=os.path.join(save_config.model_save_dir, self.save_path),
            name_prefix=self.name_prefix,
            save_replay_buffer=self.save_replay_buffer,
            verbose=self.verbose,
        )


class VideoRecorderCallbackConfigReader(BaseModel, YAMLReaderMixin):
    render_freq: int = Field(ge=1, default=100)
    save_dir: str = "ckpts"
    name_prefix: str = "rl_model"
    file_ext: str = "gif"
    deterministic: bool = False

    def to_config(self, save_config: SaveConfig) -> VideoRecorderCallbackConfig:
        return VideoRecorderCallbackConfig(
            render_freq=self.render_freq,
            save_dir=os.path.join(save_config.model_save_dir, self.save_dir),
            name_prefix=self.name_prefix,
            file_ext=self.file_ext,
            deterministic=self.deterministic,
        )


class SB3CallbackConfigReader(BaseModel, YAMLReaderMixin):
    eval_callback_config: EvalCallbackConfigReader
    ckpt_callback_config: CheckpointCallbackConfigReader = (
        CheckpointCallbackConfigReader()
    )
    video_recorder_callback_config: VideoRecorderCallbackConfigReader | None = None

    def to_config(self, save_config: SaveConfig) -> SB3CallbackConfig:
        eval_callback_config = self.eval_callback_config.to_config(
            save_config=save_config
        )
        ckpt_callback_config = self.ckpt_callback_config.to_config(
            save_config=save_config
        )
        video_recorder_callback_config = (
            self.video_recorder_callback_config.to_config(save_config=save_config)
            if self.video_recorder_callback_config
            else None
        )

        return SB3CallbackConfig(
            eval_callback_config=eval_callback_config,
            ckpt_callback_config=ckpt_callback_config,
            video_recorder_callback_config=video_recorder_callback_config,
        )


class SB3ExperimentManagerConfigReader(
    BaseModel, ConfigReader[SB3ExperimentManagerConfig], YAMLReaderMixin
):
    manager_class: str
    manager_config: dict[str, Any]
    callback_config: dict[str, Any]

    def to_config(self, run_name_suffix: str = "") -> SB3ExperimentManagerConfig:
        manager_class: type[SB3ExperimentManager] | None = get_class(self.manager_class)
        updated_manager_config = SB3ExperimentManager.add_run_name_suffix(
            self.manager_config, run_name_suffix
        )
        assert manager_class is not None, (
            f"Could not find experiment manager class for {self.manager_class}"
        )
        return SB3ExperimentManagerConfig(
            manager_class=manager_class,
            manager_config=updated_manager_config,
            callback_config=self.callback_config,
        )


class SB3ModelConfigReader(BaseModel, YAMLReaderMixin):
    """Configuration reader for SB3 model."""

    algo_config: SB3AlgorithmConfigReader = SB3AlgorithmConfigReader()
    learn_config: SB3LearnConfig = SB3LearnConfig()
    vec_config: MakeVecEnvConfigReader | None = None
    callback_config: SB3CallbackConfigReader

    def to_config(self, save_config: SaveConfig) -> SB3ModelConfig:
        return SB3ModelConfig(
            algo_config=self.algo_config.to_config(),
            learn_config=self.learn_config,
            vec_config=self.vec_config.to_config(save_config=save_config)
            if self.vec_config
            else None,
            callback_config=self.callback_config.to_config(save_config=save_config),
        )


class SB3PipelineConfigReader(
    BaseModel, ConfigReader[SB3PipelineConfig], YAMLReaderMixin
):
    """Configuration reader for SB3 pipeline."""

    device: str | int = "cuda:0"
    experiment_id: str = ""
    retrain_model: bool = False
    save_config: SaveConfigReader
    config_dir: str = "configs"
    env_config_file: str = "env_config.yaml"
    wrapper_config_file: str | None = None
    model_config_file: str = "model_config.yaml"
    experiment_manager_config: SB3ExperimentManagerConfigReader | None = None

    def to_config(self) -> SB3PipelineConfig:
        device: str = (
            self.device if isinstance(self.device, str) else f"cuda:{self.device}"
        )

        env_config: MakeEnvConfig = self._to_env_config()
        wrapper_config: WrapperConfig | None = self._to_wrapper_config()

        model_config_reader: SB3ModelConfigReader = self._to_model_config_reader()

        save_config: SaveConfig = self._to_save_config()

        model_config: SB3ModelConfig = model_config_reader.to_config(
            save_config=save_config
        )

        vec_config = model_config.vec_config
        algo_config = model_config.algo_config
        learn_config = model_config.learn_config
        callback_config = model_config.callback_config

        experiment_manager_config: SB3ExperimentManagerConfig | None = (
            self._to_manager_config()
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
            callback_config=callback_config,
            experiment_manager_config=experiment_manager_config,
        )

        return pipeline_config

    def _to_env_config(self) -> MakeEnvConfig:
        env_config: MakeEnvConfig = read_config_dict_from_yaml(
            self.config_dir, self.env_config_file, MakeEnvConfig
        )
        return env_config

    def _to_wrapper_config(self) -> WrapperConfig | None:
        if self.wrapper_config_file:
            wrapper_config_reader = read_config_dict_from_yaml(
                self.config_dir, self.wrapper_config_file, WrapperConfigReader
            )
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        else:
            return None

    def _to_model_config_reader(self) -> SB3ModelConfigReader:
        model_config_reader: SB3ModelConfigReader = read_config_dict_from_yaml(
            self.config_dir, self.model_config_file, SB3ModelConfigReader
        )

        return model_config_reader

    def _to_save_config(self) -> SaveConfig:
        model_config_reader = self._to_model_config_reader()
        return self.save_config.to_config(
            experiment_id=self.experiment_id,
            model_name_suffix="_"
            + format_large_number(model_config_reader.learn_config.total_timesteps),
        )

    def _to_manager_config(self) -> SB3ExperimentManagerConfig | None:
        if self.experiment_manager_config:
            return self.experiment_manager_config.to_config()
        return None


class SB3ReplicatePipelineConfigReader(
    BaseModel, ConfigReader[SB3ReplicatePipelineConfig], YAMLReaderMixin
):
    """Configuration reader for SB3 replicate pipeline."""

    device: str | int = "cuda:0"
    experiment_id: str = ""
    retrain_model: bool = False
    replicate_config: ReplicateConfig
    save_config: SaveConfigReader
    config_dir: str = "configs"
    env_config_file: str = "env_config.yaml"
    wrapper_config_file: str | None = None
    model_config_file: str = "model_config.yaml"
    experiment_manager_config: SB3ExperimentManagerConfigReader | None = None

    def to_config(self) -> SB3ReplicatePipelineConfig:
        replicate_pipeline_configs: list[SB3PipelineConfig] = []
        for rep_id in range(self.replicate_config.num_replicates):
            device: str = (
                self.device if isinstance(self.device, str) else f"cuda:{self.device}"
            )

            env_config: MakeEnvConfig = self._to_env_config()
            wrapper_config: WrapperConfig | None = self._to_wrapper_config()

            model_config_reader: SB3ModelConfigReader = self._to_model_config_reader()

            save_config: SaveConfig = self._to_save_config(rep_id)

            model_config: SB3ModelConfig = model_config_reader.to_config(
                save_config=save_config
            )

            vec_config = model_config.vec_config
            algo_config = model_config.algo_config
            learn_config = model_config.learn_config
            callback_config = model_config.callback_config

            experiment_manager_config: SB3ExperimentManagerConfig | None = (
                self._to_manager_config(rep_id=rep_id)
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
                callback_config=callback_config,
                experiment_manager_config=experiment_manager_config,
            )

            replicate_pipeline_configs.append(pipeline_config)

        pipeline_config = SB3ReplicatePipelineConfig(
            replicate_config=self.replicate_config,
            ind_pipeline_configs=replicate_pipeline_configs,
        )

        return pipeline_config

    def _to_env_config(self) -> MakeEnvConfig:
        env_config: MakeEnvConfig = read_config_dict_from_yaml(
            self.config_dir, self.env_config_file, MakeEnvConfig
        )
        return env_config

    def _to_wrapper_config(self) -> WrapperConfig | None:
        if self.wrapper_config_file:
            wrapper_config_reader = read_config_dict_from_yaml(
                self.config_dir, self.wrapper_config_file, WrapperConfigReader
            )
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        return None

    def _to_model_config_reader(self) -> SB3ModelConfigReader:
        model_config_reader: SB3ModelConfigReader = read_config_dict_from_yaml(
            self.config_dir, self.model_config_file, SB3ModelConfigReader
        )

        return model_config_reader

    def _to_save_config(self, rep_id: int) -> SaveConfig:
        model_config_reader = self._to_model_config_reader()
        return self.save_config.to_config(
            experiment_id=self.experiment_id,
            model_name_suffix="_"
            + format_large_number(model_config_reader.learn_config.total_timesteps),
            replicate_signature=self.replicate_config.replicate_signature.format(
                rep_id=rep_id + self.replicate_config.replicate_start_id
            ),
        )

    def _to_manager_config(self, rep_id: int) -> SB3ExperimentManagerConfig | None:
        replicate_signature: str = (
            "_"
            + self.replicate_config.replicate_signature.format(
                rep_id=rep_id + self.replicate_config.replicate_start_id
            )
        )
        if self.experiment_manager_config:
            return self.experiment_manager_config.to_config(
                run_name_suffix=replicate_signature
            )
        return None
