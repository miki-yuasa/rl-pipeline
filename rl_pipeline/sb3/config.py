"""Configuration models for SB3 training, evaluation, and hyperparameter tuning.

This module defines strongly typed Pydantic models used by the SB3 pipeline.
It includes model/training/callback settings and Optuna-related tuning settings.
"""

from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_pipeline.core import ReplicateConfig, SaveConfig
from rl_pipeline.gymnasium.config import MakeEnvConfig, WrapperConfig

from .callback import (
    CheckpointCallbackConfig,
    EvalCallbackConfig,
    VideoRecorderCallbackConfig,
)
from .experiment import SB3ExperimentManager


def class_to_string(cls: type) -> str:
    """Serialize a Python class to a fully qualified import path.

    Parameters
    ----------
    cls : type
        Class object to serialize.

    Returns
    -------
    class_path : str
        Dotted module path in the form ``package.module.ClassName``.
    """
    return f"{cls.__module__}.{cls.__name__}"


class MakeVecEnvConfig(BaseModel):
    """Configuration for vectorized environment creation.

    Attributes
    ----------
    n_envs : int
        Number of parallel environments.
    seed : int | None
        Random seed used during environment construction.
    start_index : int
        Start index for seeded env IDs.
    monitor_dir : str | None
        Directory where monitor files are written.
    vec_env_cls : type[SubprocVecEnv] | type[DummyVecEnv] | None
        Vectorized environment implementation.
    vec_env_kwargs : dict[str, Any] | None
        Extra kwargs passed to the vec env class.
    monitor_kwargs : dict[str, Any] | None
        Extra kwargs for environment monitoring wrappers.
    """

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
        """Serialize the vectorized env class for JSON output.

        Parameters
        ----------
        vec_env_cls : type[SubprocVecEnv | DummyVecEnv]
            Class to serialize.

        Returns
        -------
        vec_env_cls_path : str
            Import path string for the class.
        """
        return class_to_string(vec_env_cls)


class SB3AlgorithmConfig(BaseModel):
    """Algorithm class and constructor kwargs for SB3 model creation.

    Attributes
    ----------
    algorithm : type[BaseAlgorithm]
        SB3-compatible algorithm class.
    algo_kwargs : dict[str, Any]
        Keyword arguments passed to the algorithm constructor.
    """

    algorithm: type[BaseAlgorithm]
    algo_kwargs: dict[str, Any]

    # For json serialization, `algorithm` should be in a string (e.b. "stable_baselines.PPO")
    @field_serializer("algorithm", when_used="json")
    def serialize_algorithm(self, algorithm: type[BaseAlgorithm]) -> str:
        """Serialize the algorithm class for JSON output.

        Parameters
        ----------
        algorithm : type[BaseAlgorithm]
            Algorithm class to serialize.

        Returns
        -------
        algorithm_path : str
            Import path string for the algorithm class.
        """
        return class_to_string(algorithm)


class SB3LearnConfig(BaseModel):
    """Training-loop arguments forwarded to ``model.learn``.

    Attributes
    ----------
    total_timesteps : int
        Number of environment timesteps for training.
    log_interval : int
        Logging frequency used by SB3.
    tb_log_name : str
        TensorBoard run name.
    reset_num_timesteps : bool
        Whether to reset timestep counter before learning.
    progress_bar : bool
        Whether to display a progress bar during training.
    """

    total_timesteps: int = Field(ge=1, default=1_000_000)
    # callback: MaybeCallback = None
    log_interval: int = Field(ge=0, default=1)
    tb_log_name: str = "run"
    reset_num_timesteps: bool = True
    progress_bar: bool = False

    # allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ArbitraryCallbackConfig(BaseModel):
    """Configuration for attaching user-defined SB3 callbacks.

    Attributes
    ----------
    callback_class : type[BaseCallback]
        Callback class to instantiate.
    callback_kwargs : dict[str, Any]
        Initialization kwargs for ``callback_class``.
    """

    callback_class: type[BaseCallback]
    callback_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("callback_class", when_used="json")
    def serialize_callback_class(self, callback_class: type[BaseCallback]) -> str:
        """Serialize callback class for JSON output.

        Parameters
        ----------
        callback_class : type[BaseCallback]
            Callback class to serialize.

        Returns
        -------
        callback_class_path : str
            Import path string for the callback class.
        """
        return class_to_string(callback_class)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SB3CallbackConfig(BaseModel):
    """Grouped callback settings used by the SB3 pipeline.

    Attributes
    ----------
    eval_callback_config : EvalCallbackConfig
        Evaluation callback configuration.
    ckpt_callback_config : CheckpointCallbackConfig
        Checkpoint callback configuration.
    video_recorder_callback_config : VideoRecorderCallbackConfig | None
        Optional video recording callback configuration.
    arbitrary_callback_configs : list[ArbitraryCallbackConfig]
        Additional user-defined callbacks.
    """

    eval_callback_config: EvalCallbackConfig
    ckpt_callback_config: CheckpointCallbackConfig
    video_recorder_callback_config: VideoRecorderCallbackConfig | None = None
    arbitrary_callback_configs: list[ArbitraryCallbackConfig] = Field(
        default_factory=list
    )


class SB3ExperimentManagerConfig(BaseModel):
    """Configuration for experiment manager integration.

    Attributes
    ----------
    manager_class : type[SB3ExperimentManager]
        Experiment manager implementation class.
    manager_config : dict[str, Any]
        Manager initialization arguments.
    callback_config : dict[str, Any]
        Manager-specific callback configuration.
    """

    manager_class: type[SB3ExperimentManager]
    manager_config: dict[str, Any]
    callback_config: dict[str, Any]

    @field_serializer("manager_class", when_used="json")
    def serialize_manager_class(self, manager_class: type[SB3ExperimentManager]) -> str:
        """Serialize manager class for JSON output.

        Parameters
        ----------
        manager_class : type[SB3ExperimentManager]
            Manager class to serialize.

        Returns
        -------
        manager_class_path : str
            Import path string for the manager class.
        """
        return manager_class.__module__ + "." + manager_class.__name__

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SB3OptunaDashboardConfig(BaseModel):
    """Runtime settings for optional ``optuna-dashboard`` launch.

    Attributes
    ----------
    launch : bool
        Whether to start the dashboard subprocess automatically.
    host : str | None
        Optional host passed to dashboard CLI.
    port : int | None
        Optional port passed to dashboard CLI.
    """

    launch: bool = False
    host: str | None = None
    port: int | None = Field(default=None, ge=1)


class SB3OptunaParamConfig(BaseModel):
    """Declarative search-space specification for one tunable parameter.

    Attributes
    ----------
    name : str
        Optuna trial parameter name.
    suggest_type : Literal["float", "int", "categorical", "pow2_int"]
        Suggestion strategy used to sample this parameter.
    target : str | None
        Destination key in algorithm kwargs (supports dotted paths).
    low : float | int | None
        Lower bound for numeric suggestion types.
    high : float | int | None
        Upper bound for numeric suggestion types.
    step : float | int | None
        Step size for integer/float suggestion types.
    log : bool
        Whether to sample numeric values in log space.
    choices : list[Any] | None
        Choice list for categorical parameters.
    one_minus : bool
        Whether to transform sampled value as ``1.0 - value``.
    value_mapping : dict[str, Any] | None
        Optional mapping from categorical values to runtime objects.
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


class SB3OptunaConfig(BaseModel):
    """Optuna optimization settings used by :meth:`SB3Pipeline.optimize`.

    Attributes
    ----------
    storage_url : str | None
        Optuna storage backend URL.
    study_name : str | None
        Optional study name in storage.
    direction : Literal["maximize", "minimize"]
        Optimization direction for objective value.
    n_trials : int
        Number of trials to run.
    timeout : int | None
        Optional timeout (seconds) for optimization.
    n_jobs : int
        Number of parallel workers used for optimization.
    parallel_backend : Literal["process", "thread"]
        Parallel execution backend. ``"process"`` is default.
    n_startup_trials : int
        Number of startup trials before TPE-based decisions.
    n_warmup_steps : int
        Number of evaluation steps before pruning can start.
    n_evaluations : int
        Number of evaluations during each trial.
    n_eval_episodes : int
        Number of episodes per evaluation.
    deterministic_eval : bool
        Whether evaluation uses deterministic actions.
    total_timesteps : int | None
        Optional per-trial timestep override.
    sample_params_fn : Callable[[Any], dict[str, Any]] | None
        Optional custom sampling callable.
    tune_params : list[SB3OptunaParamConfig]
        Declarative search-space definitions.
    dashboard : SB3OptunaDashboardConfig
        Dashboard launch settings.
    """

    storage_url: str | None = None
    study_name: str | None = None
    direction: Literal["maximize", "minimize"] = "maximize"
    n_trials: int = Field(ge=1, default=50)
    timeout: int | None = Field(default=None, ge=1)
    n_jobs: int = Field(ge=1, default=1)
    parallel_backend: Literal["process", "thread"] = "process"
    n_startup_trials: int = Field(ge=0, default=5)
    n_warmup_steps: int = Field(ge=0, default=0)
    n_evaluations: int = Field(ge=1, default=2)
    n_eval_episodes: int = Field(ge=1, default=3)
    deterministic_eval: bool = False
    total_timesteps: int | None = Field(default=None, ge=1)
    sample_params_fn: Callable[[Any], dict[str, Any]] | None = None
    tune_params: list[SB3OptunaParamConfig] = Field(default_factory=list)
    dashboard: SB3OptunaDashboardConfig = SB3OptunaDashboardConfig()

    @field_serializer("sample_params_fn", when_used="json")
    def serialize_sample_params_fn(
        self, sample_params_fn: Callable[[Any], dict[str, Any]] | None
    ) -> str | None:
        """Serialize custom sampling callable for JSON output.

        Parameters
        ----------
        sample_params_fn : Callable[[Any], dict[str, Any]] | None
            Custom sampling callable.

        Returns
        -------
        sample_params_fn_path : str | None
            Import path for callable or ``None`` if unset.
        """
        if sample_params_fn is None:
            return None
        return f"{sample_params_fn.__module__}.{sample_params_fn.__name__}"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SB3PipelineConfig(BaseModel):
    """Top-level configuration for a single SB3 pipeline run.

    Attributes
    ----------
    device : str
        Target compute device string (for example ``cuda:0`` or ``cpu``).
    experiment_id : str
        Identifier used to organize saved artifacts.
    retrain_model : bool
        Whether existing saved models should be retrained.
    save_config : SaveConfig
        Output path configuration.
    env_config : MakeEnvConfig
        Base environment construction config.
    wrapper_config : WrapperConfig | None
        Optional single-env wrapper configuration.
    vec_config : MakeVecEnvConfig | None
        Optional vectorized environment configuration.
    algo_config : SB3AlgorithmConfig
        Algorithm class and constructor kwargs.
    learn_config : SB3LearnConfig
        ``model.learn`` training settings.
    callback_config : SB3CallbackConfig
        Callback settings.
    experiment_manager_config : SB3ExperimentManagerConfig | None
        Optional run manager integration settings.
    optuna_config : SB3OptunaConfig | None
        Optional optimization configuration for hyperparameter search.
    """

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
    optuna_config: SB3OptunaConfig | None = None


class SB3ModelConfig(BaseModel):
    """Bundled model-centric configuration loaded from model config files.

    Attributes
    ----------
    algo_config : SB3AlgorithmConfig
        Algorithm class and kwargs.
    learn_config : SB3LearnConfig
        Training loop settings.
    vec_config : MakeVecEnvConfig | None
        Optional vectorized environment settings.
    callback_config : SB3CallbackConfig
        Callback configuration.
    """

    algo_config: SB3AlgorithmConfig
    learn_config: SB3LearnConfig
    vec_config: MakeVecEnvConfig | None = None
    callback_config: SB3CallbackConfig


class SB3ReplicatePipelineConfig(BaseModel):
    """Configuration for running multiple independent SB3 pipeline replicas.

    Attributes
    ----------
    replicate_config : ReplicateConfig
        Replica generation policy.
    ind_pipeline_configs : list[SB3PipelineConfig]
        Fully resolved configs for each replica run.
    """

    replicate_config: ReplicateConfig
    ind_pipeline_configs: list[SB3PipelineConfig]
