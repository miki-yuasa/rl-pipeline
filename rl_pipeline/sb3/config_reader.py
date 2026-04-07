"""YAML-to-runtime configuration readers for the SB3 pipeline.

Readers in this module deserialize YAML-friendly structures and resolve
runtime Python objects (classes/callables) used by the SB3 pipeline.
"""

import importlib.util
import os
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
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
    ArbitraryCallbackConfig,
    MakeVecEnvConfig,
    SB3AlgorithmConfig,
    SB3CallbackConfig,
    SB3ExperimentManagerConfig,
    SB3LearnConfig,
    SB3ModelConfig,
    SB3OptunaConfig,
    SB3OptunaDashboardConfig,
    SB3OptunaParamConfig,
    SB3PipelineConfig,
    SB3ReplicatePipelineConfig,
)
from .experiment.base import SB3ExperimentManager


class SB3AlgorithmConfigReader(
    BaseModel, ConfigReader[SB3AlgorithmConfig], YAMLReaderMixin
):
    """Reader for SB3 algorithm config.

    Attributes
    ----------
    algorithm : str
        Algorithm name or fully qualified class path.
    algo_kwargs : dict[str, Any]
        Constructor kwargs for the algorithm.
    """

    algorithm: str = "PPO"
    algo_kwargs: dict[str, Any] = {}

    def to_config(self) -> SB3AlgorithmConfig:
        """Resolve and build :class:`SB3AlgorithmConfig`.

        Returns
        -------
        config : SB3AlgorithmConfig
            Runtime algorithm config with resolved class object.
        """
        algo_class: type[BaseAlgorithm] | None = None

        if "." in self.algorithm:
            algo_class = get_class(self.algorithm)
        else:
            try:
                algo_class = get_class("stable_baselines3." + self.algorithm)
            except (ModuleNotFoundError, ImportError, AttributeError):
                if importlib.util.find_spec("sb3_contrib") is not None:
                    try:
                        algo_class = get_class("sb3_contrib." + self.algorithm)
                    except (ModuleNotFoundError, ImportError, AttributeError):
                        algo_class = None

        assert algo_class is not None, (
            "Could not find algorithm class for "
            f"{self.algorithm} in stable_baselines3 or sb3_contrib"
        )

        return SB3AlgorithmConfig(
            algorithm=algo_class,
            algo_kwargs=self.algo_kwargs,
        )


class SB3LearnConfigReader(BaseModel, ConfigReader[SB3LearnConfig], YAMLReaderMixin):
    """Reader for training-loop settings."""

    total_timesteps: int = Field(ge=1, default=1_000_000)
    log_interval: int = Field(ge=0, default=100)
    tb_log_name: str = "run"
    reset_num_timesteps: bool = True
    progress_bar: bool = False

    def to_config(self) -> SB3LearnConfig:
        """Build :class:`SB3LearnConfig` from reader fields.

        Returns
        -------
        config : SB3LearnConfig
            Runtime training settings.
        """
        return SB3LearnConfig(
            total_timesteps=self.total_timesteps,
            log_interval=self.log_interval,
            tb_log_name=self.tb_log_name,
            reset_num_timesteps=self.reset_num_timesteps,
            progress_bar=self.progress_bar,
        )


class MakeVecEnvConfigReader(BaseModel, YAMLReaderMixin):
    """Reader for vectorized environment settings."""

    n_envs: int = Field(ge=1, default=1)
    seed: int | None = None
    start_index: int = Field(ge=0, default=0)
    vec_env_cls: str | None = "SubprocVecEnv"
    vec_env_kwargs: dict[str, Any] | None = None
    monitor_kwargs: dict[str, Any] | None = None

    def to_config(self, save_config: SaveConfig) -> MakeVecEnvConfig:
        """Build :class:`MakeVecEnvConfig` with resolved vec env class.

        Parameters
        ----------
        save_config : SaveConfig
            Save paths used to derive monitor output directories.

        Returns
        -------
        config : MakeVecEnvConfig
            Runtime vectorized environment configuration.
        """
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
    """Reader for evaluation callback settings."""

    eval_freq: int = Field(ge=0)
    n_eval_episodes: int = Field(ge=1)
    log_path: str = "eval"
    deterministic: bool = False
    render: bool = False

    def to_config(self, save_config: SaveConfig) -> EvalCallbackConfig:
        """Build :class:`EvalCallbackConfig`.

        Parameters
        ----------
        save_config : SaveConfig
            Save paths used to build evaluation log paths.

        Returns
        -------
        config : EvalCallbackConfig
            Runtime eval callback config.
        """
        return EvalCallbackConfig(
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            best_model_save_path=save_config.model_save_dir,
            log_path=os.path.join(save_config.model_save_dir, self.log_path),
            deterministic=self.deterministic,
            render=self.render,
        )


class CheckpointCallbackConfigReader(BaseModel, YAMLReaderMixin):
    """Reader for checkpoint callback settings."""

    save_freq: int = Field(ge=1, default=100)
    save_path: str = "ckpts"
    name_prefix: str = "ckpt"
    save_replay_buffer: bool = False
    verbose: int = 0

    def to_config(self, save_config: SaveConfig) -> CheckpointCallbackConfig:
        """Build :class:`CheckpointCallbackConfig`.

        Parameters
        ----------
        save_config : SaveConfig
            Save paths used to derive checkpoint directory.

        Returns
        -------
        config : CheckpointCallbackConfig
            Runtime checkpoint callback config.
        """
        return CheckpointCallbackConfig(
            save_freq=self.save_freq,
            save_path=os.path.join(save_config.model_save_dir, self.save_path),
            name_prefix=self.name_prefix,
            save_replay_buffer=self.save_replay_buffer,
            verbose=self.verbose,
        )


class VideoRecorderCallbackConfigReader(BaseModel, YAMLReaderMixin):
    """Reader for video recorder callback settings."""

    render_freq: int = Field(ge=1, default=100)
    save_dir: str = "ckpts"
    name_prefix: str = "rl_model"
    file_ext: str = "gif"
    deterministic: bool = False

    def to_config(self, save_config: SaveConfig) -> VideoRecorderCallbackConfig:
        """Build :class:`VideoRecorderCallbackConfig`.

        Parameters
        ----------
        save_config : SaveConfig
            Save paths used to derive animation output directory.

        Returns
        -------
        config : VideoRecorderCallbackConfig
            Runtime video recorder callback config.
        """
        return VideoRecorderCallbackConfig(
            render_freq=self.render_freq,
            save_dir=os.path.join(save_config.model_save_dir, self.save_dir),
            name_prefix=self.name_prefix,
            file_ext=self.file_ext,
            deterministic=self.deterministic,
        )


class ArbitraryCallbackConfigReader(BaseModel, YAMLReaderMixin):
    """Reader for user-defined callback entries."""

    callback_class: str
    callback_kwargs: dict[str, Any] = Field(default_factory=dict)

    def to_config(self) -> ArbitraryCallbackConfig:
        """Resolve callback class and build runtime callback config.

        Returns
        -------
        config : ArbitraryCallbackConfig
            Runtime callback class/kwargs pair.
        """
        callback_class: type | None = get_class(self.callback_class)
        assert callback_class is not None, (
            f"Could not find callback class for {self.callback_class}"
        )
        assert isinstance(callback_class, type) and issubclass(
            callback_class, BaseCallback
        ), f"Callback class {self.callback_class} must inherit from BaseCallback"

        return ArbitraryCallbackConfig(
            callback_class=callback_class,
            callback_kwargs=self.callback_kwargs,
        )


class SB3CallbackConfigReader(BaseModel, YAMLReaderMixin):
    """Reader for grouped callback configuration."""

    eval_callback_config: EvalCallbackConfigReader
    ckpt_callback_config: CheckpointCallbackConfigReader = (
        CheckpointCallbackConfigReader()
    )
    video_recorder_callback_config: VideoRecorderCallbackConfigReader | None = None
    arbitrary_callback_configs: list[ArbitraryCallbackConfigReader] = Field(
        default_factory=list
    )

    def to_config(self, save_config: SaveConfig) -> SB3CallbackConfig:
        """Build :class:`SB3CallbackConfig` and nested callback configs.

        Parameters
        ----------
        save_config : SaveConfig
            Save path configuration used by nested callbacks.

        Returns
        -------
        config : SB3CallbackConfig
            Runtime callback configuration bundle.
        """
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
        arbitrary_callback_configs = [
            callback_config.to_config()
            for callback_config in self.arbitrary_callback_configs
        ]

        return SB3CallbackConfig(
            eval_callback_config=eval_callback_config,
            ckpt_callback_config=ckpt_callback_config,
            video_recorder_callback_config=video_recorder_callback_config,
            arbitrary_callback_configs=arbitrary_callback_configs,
        )


class SB3ExperimentManagerConfigReader(
    BaseModel, ConfigReader[SB3ExperimentManagerConfig], YAMLReaderMixin
):
    """Reader for SB3 experiment manager settings."""

    manager_class: str
    manager_config: dict[str, Any]
    callback_config: dict[str, Any]

    def to_config(self, run_name_suffix: str = "") -> SB3ExperimentManagerConfig:
        """Resolve manager class and build runtime manager config.

        Parameters
        ----------
        run_name_suffix : str, optional
            Suffix appended to manager run names, by default "".

        Returns
        -------
        config : SB3ExperimentManagerConfig
            Runtime manager configuration.
        """
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


class SB3OptunaDashboardConfigReader(BaseModel, YAMLReaderMixin):
    """Reader for optuna-dashboard launch settings."""

    launch: bool = False
    host: str | None = None
    port: int | None = Field(default=None, ge=1)

    def to_config(self) -> SB3OptunaDashboardConfig:
        """Build :class:`SB3OptunaDashboardConfig`.

        Returns
        -------
        config : SB3OptunaDashboardConfig
            Runtime dashboard launch settings.
        """
        return SB3OptunaDashboardConfig(
            launch=self.launch,
            host=self.host,
            port=self.port,
        )


class SB3OptunaParamConfigReader(BaseModel, YAMLReaderMixin):
    """Reader for one declarative Optuna search-space entry.

    Notes
    -----
    ``value_mapping`` values that look like dotted import paths are resolved
    to runtime objects when possible.
    """

    name: str
    suggest_type: Literal["float", "int", "categorical", "pow2_int"]
    target: str | None = None
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[Any] | None = None
    one_minus: bool = False
    value_mapping: dict[str, Any] | None = None

    def to_config(self) -> SB3OptunaParamConfig:
        """Build :class:`SB3OptunaParamConfig`.

        Returns
        -------
        config : SB3OptunaParamConfig
            Runtime parameter search-space configuration.
        """
        resolved_value_mapping: dict[str, Any] | None = None
        if self.value_mapping is not None:
            resolved_value_mapping = {}
            for key, value in self.value_mapping.items():
                if isinstance(value, str) and "." in value:
                    try:
                        resolved_value_mapping[key] = get_class(value)
                    except (AttributeError, ImportError, ModuleNotFoundError):
                        resolved_value_mapping[key] = value
                else:
                    resolved_value_mapping[key] = value

        return SB3OptunaParamConfig(
            name=self.name,
            suggest_type=self.suggest_type,
            target=self.target,
            low=self.low,
            high=self.high,
            step=self.step,
            log=self.log,
            choices=self.choices,
            one_minus=self.one_minus,
            value_mapping=resolved_value_mapping,
        )


class SB3OptunaConfigReader(BaseModel, YAMLReaderMixin):
    """Reader for Optuna optimization settings."""

    storage_url: str | None = None
    study_name: str | None = None
    direction: Literal["maximize", "minimize"] = "maximize"
    n_trials: int = Field(ge=1, default=50)
    timeout: int | None = Field(default=None, ge=1)
    n_jobs: int = Field(ge=1, default=1)
    n_startup_trials: int = Field(ge=0, default=5)
    n_warmup_steps: int = Field(ge=0, default=0)
    n_evaluations: int = Field(ge=1, default=2)
    n_eval_episodes: int = Field(ge=1, default=3)
    deterministic_eval: bool = True
    total_timesteps: int | None = Field(default=None, ge=1)
    sample_params_fn: str | None = None
    tune_params: list[SB3OptunaParamConfigReader] = Field(default_factory=list)
    dashboard: SB3OptunaDashboardConfigReader = SB3OptunaDashboardConfigReader()

    def to_config(self) -> SB3OptunaConfig:
        """Build :class:`SB3OptunaConfig`.

        Returns
        -------
        config : SB3OptunaConfig
            Runtime Optuna optimization configuration.
        """
        sample_params_fn = None
        if self.sample_params_fn is not None:
            sample_params_fn = get_class(self.sample_params_fn)
            assert callable(sample_params_fn), (
                f"sample_params_fn must be callable: {self.sample_params_fn}"
            )

        return SB3OptunaConfig(
            storage_url=self.storage_url,
            study_name=self.study_name,
            direction=self.direction,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            n_startup_trials=self.n_startup_trials,
            n_warmup_steps=self.n_warmup_steps,
            n_evaluations=self.n_evaluations,
            n_eval_episodes=self.n_eval_episodes,
            deterministic_eval=self.deterministic_eval,
            total_timesteps=self.total_timesteps,
            sample_params_fn=sample_params_fn,
            tune_params=[param.to_config() for param in self.tune_params],
            dashboard=self.dashboard.to_config(),
        )


class SB3ModelConfigReader(BaseModel, YAMLReaderMixin):
    """Reader for model-level algorithm/training/callback configuration."""

    algo_config: SB3AlgorithmConfigReader = SB3AlgorithmConfigReader()
    learn_config: SB3LearnConfig = SB3LearnConfig()
    vec_config: MakeVecEnvConfigReader | None = None
    callback_config: SB3CallbackConfigReader

    def to_config(self, save_config: SaveConfig) -> SB3ModelConfig:
        """Build :class:`SB3ModelConfig`.

        Parameters
        ----------
        save_config : SaveConfig
            Save configuration used by vectorized env/callback readers.

        Returns
        -------
        config : SB3ModelConfig
            Runtime model configuration.
        """
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
    """Reader for top-level SB3 pipeline configuration."""

    device: str | int = "cuda:0"
    experiment_id: str = ""
    retrain_model: bool = False
    save_config: SaveConfigReader
    config_dir: str = "configs"
    env_config_file: str = "env_config.yaml"
    wrapper_config_file: str | None = None
    model_config_file: str = "model_config.yaml"
    experiment_manager_config: SB3ExperimentManagerConfigReader | None = None
    optuna_config: SB3OptunaConfigReader | None = None

    def to_config(self) -> SB3PipelineConfig:
        """Build :class:`SB3PipelineConfig` from YAML-linked sub-configs.

        Returns
        -------
        config : SB3PipelineConfig
            Fully resolved runtime pipeline configuration.
        """
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

        optuna_config: SB3OptunaConfig | None = (
            self.optuna_config.to_config() if self.optuna_config else None
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
            optuna_config=optuna_config,
        )

        return pipeline_config

    def _to_env_config(self) -> MakeEnvConfig:
        """Load and parse environment config file.

        Returns
        -------
        env_config : MakeEnvConfig
            Runtime environment configuration.
        """
        env_config: MakeEnvConfig = read_config_dict_from_yaml(
            self.config_dir, self.env_config_file, MakeEnvConfig
        )
        return env_config

    def _to_wrapper_config(self) -> WrapperConfig | None:
        """Load and parse optional wrapper config file.

        Returns
        -------
        wrapper_config : WrapperConfig | None
            Wrapper configuration or ``None`` when not configured.
        """
        if self.wrapper_config_file:
            wrapper_config_reader = read_config_dict_from_yaml(
                self.config_dir, self.wrapper_config_file, WrapperConfigReader
            )
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        else:
            return None

    def _to_model_config_reader(self) -> SB3ModelConfigReader:
        """Load model config reader from YAML file.

        Returns
        -------
        model_config_reader : SB3ModelConfigReader
            Parsed model config reader instance.
        """
        model_config_reader: SB3ModelConfigReader = read_config_dict_from_yaml(
            self.config_dir, self.model_config_file, SB3ModelConfigReader
        )

        return model_config_reader

    def _to_save_config(self, replicate_signature: str = "") -> SaveConfig:
        """Build save config for this pipeline instance.

        Parameters
        ----------
        replicate_signature : str, optional
            Replica suffix used for path disambiguation, by default "".

        Returns
        -------
        save_config : SaveConfig
            Runtime save path configuration.
        """
        model_config_reader = self._to_model_config_reader()
        return self.save_config.to_config(
            experiment_id=self.experiment_id,
            model_name_suffix=format_large_number(
                model_config_reader.learn_config.total_timesteps
            ),
            replicate_signature=replicate_signature,
        )

    def _to_manager_config(
        self, replicate_signature: str = ""
    ) -> SB3ExperimentManagerConfig | None:
        """Build optional experiment manager config.

        Parameters
        ----------
        replicate_signature : str, optional
            Replica suffix forwarded to manager config, by default "".

        Returns
        -------
        manager_config : SB3ExperimentManagerConfig | None
            Runtime manager config or ``None``.
        """
        if self.experiment_manager_config:
            return self.experiment_manager_config.to_config(
                run_name_suffix=replicate_signature
            )
        return None


SB3PipelineConfigReaderType = TypeVar(
    "SB3PipelineConfigReaderType", bound=SB3PipelineConfigReader
)


class SB3ReplicatePipelineConfigReader(
    BaseModel,
    ConfigReader[SB3ReplicatePipelineConfig],
    YAMLReaderMixin,
    Generic[SB3PipelineConfigReaderType],
):
    """Reader for replicate SB3 pipeline configuration."""

    replicate_config: ReplicateConfig
    single_pipeline_config: SB3PipelineConfigReaderType

    def to_config(self) -> SB3ReplicatePipelineConfig:
        """Generate per-replica pipeline configs.

        Returns
        -------
        config : SB3ReplicatePipelineConfig
            Runtime replicate pipeline configuration.
        """
        replicate_pipeline_configs: list[SB3PipelineConfig] = []
        for rep_id in range(self.replicate_config.num_replicates):
            device: str = (
                self.single_pipeline_config.device
                if isinstance(self.single_pipeline_config.device, str)
                else f"cuda:{self.single_pipeline_config.device}"
            )

            env_config: MakeEnvConfig = self.single_pipeline_config._to_env_config()
            wrapper_config: WrapperConfig | None = (
                self.single_pipeline_config._to_wrapper_config()
            )

            model_config_reader: SB3ModelConfigReader = (
                self.single_pipeline_config._to_model_config_reader()
            )

            rep_signature = self.replicate_config.replicate_signature.format(
                rep_id=rep_id + self.replicate_config.replicate_start_id
            )
            save_config: SaveConfig = self.single_pipeline_config._to_save_config(
                replicate_signature=rep_signature
            )

            model_config: SB3ModelConfig = model_config_reader.to_config(
                save_config=save_config
            )

            experiment_manager_config: SB3ExperimentManagerConfig | None = (
                self.single_pipeline_config._to_manager_config(
                    replicate_signature=rep_signature
                )
            )

            pipeline_config = SB3PipelineConfig(
                device=device,
                experiment_id=self.single_pipeline_config.experiment_id,
                retrain_model=self.single_pipeline_config.retrain_model,
                save_config=save_config,
                env_config=env_config,
                wrapper_config=wrapper_config,
                vec_config=model_config.vec_config,
                algo_config=model_config.algo_config,
                learn_config=model_config.learn_config,
                callback_config=model_config.callback_config,
                experiment_manager_config=experiment_manager_config,
            )

            replicate_pipeline_configs.append(pipeline_config)

        pipeline_config = SB3ReplicatePipelineConfig(
            replicate_config=self.replicate_config,
            ind_pipeline_configs=replicate_pipeline_configs,
        )

        return pipeline_config
