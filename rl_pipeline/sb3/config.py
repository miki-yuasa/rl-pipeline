from __future__ import annotations

"""Configuration models for SB3 training, evaluation, and replicate pipelines."""

import os
from copy import deepcopy
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_pipeline.core import ReplicateConfig, SaveConfig
from rl_pipeline.core.config import YAMLReaderMixin
from rl_pipeline.core.utils.io import format_large_number, get_class
from rl_pipeline.gymnasium.config import (
    MakeEnvConfig,
    WrapperConfig,
    load_wrapper_config,
)

from .algo_config import (
    SB3AlgorithmConfig,
    SB3LearnConfig,
    class_to_string,
)
from .callback_config import (
    ArbitraryCallbackConfig,
    SB3CallbackConfig,
)
from .experiment.base import SB3ExperimentManager
from .optuna_config import (
    SB3OptunaConfig,
    SB3OptunaDashboardConfig,
    SB3OptunaParamConfig,
)


class MakeVecEnvConfig(BaseModel, YAMLReaderMixin):
    """Configuration for vectorized environment creation."""

    n_envs: int = Field(ge=1, default=1)
    seed: int | None = None
    start_index: int = Field(ge=0, default=0)
    monitor_dir: str | None = None
    vec_env_cls: type[SubprocVecEnv | DummyVecEnv] | None = SubprocVecEnv
    vec_env_kwargs: dict[str, Any] | None = None
    monitor_kwargs: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("vec_env_cls", mode="before")
    def _resolve_vec_env_cls(
        cls, val: Any
    ) -> type[SubprocVecEnv | DummyVecEnv] | None:
        if isinstance(val, str):
            if val == "SubprocVecEnv":
                return SubprocVecEnv
            if val == "DummyVecEnv":
                return DummyVecEnv
            if "." in val:
                resolved = get_class(val)
                return resolved
        return val

    @field_serializer("vec_env_cls", when_used="json")
    def serialize_vec_env_cls(
        self, vec_env_cls: type[SubprocVecEnv | DummyVecEnv] | None
    ) -> str | None:
        return class_to_string(vec_env_cls) if vec_env_cls else None


class SB3ExperimentManagerConfig(BaseModel, YAMLReaderMixin):
    """Configuration for experiment manager integration."""

    manager_class: type[SB3ExperimentManager]
    manager_config: dict[str, Any] = Field(default_factory=dict)
    callback_config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("manager_class", mode="before")
    def _resolve_manager_class(cls, val: Any) -> type[SB3ExperimentManager]:
        if isinstance(val, str):
            resolved = get_class(val)
            return resolved
        return val

    @field_serializer("manager_class", when_used="json")
    def serialize_manager_class(
        self, manager_class: type[SB3ExperimentManager]
    ) -> str:
        return class_to_string(manager_class)


class SB3ModelConfig(BaseModel, YAMLReaderMixin):
    """Bundled model-centric configuration loaded from model config files."""

    algo_config: SB3AlgorithmConfig
    learn_config: SB3LearnConfig = Field(default_factory=SB3LearnConfig)
    vec_config: MakeVecEnvConfig | None = None
    callback_config: SB3CallbackConfig

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class SB3PipelineConfig(BaseModel, YAMLReaderMixin):
    """Top-level configuration for a single SB3 pipeline run."""

    device: str = "cuda:0"
    experiment_id: str = ""
    retrain_model: bool = False
    save_config: SaveConfig = Field(default_factory=SaveConfig)
    env_config: MakeEnvConfig | None = None
    wrapper_config: WrapperConfig | None = None
    vec_config: MakeVecEnvConfig | None = None
    algo_config: SB3AlgorithmConfig | None = None
    learn_config: SB3LearnConfig = Field(default_factory=SB3LearnConfig)
    callback_config: SB3CallbackConfig | None = None
    experiment_manager_config: SB3ExperimentManagerConfig | None = None
    optuna_config: SB3OptunaConfig | None = None

    config_dir: str | None = None
    env_config_file: str | None = None
    wrapper_class: type[Any] | str | None = None
    wrapper_config_file: str | None = None
    wrapper_kwargs: dict[str, Any] | None = None
    model_config_file: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("device", mode="before")
    def _normalize_device(cls, val: Any) -> str:
        if isinstance(val, int):
            return f"cuda:{val}"
        return str(val)

    def resolve(self, replicate_signature: str = "") -> Self:
        """Resolve nested file references and propagate artifact paths."""
        if self.env_config is None and self.env_config_file:
            env_path = (
                os.path.join(self.config_dir, self.env_config_file)
                if self.config_dir
                else self.env_config_file
            )
            self.env_config = MakeEnvConfig.from_yaml(env_path)

        if self.wrapper_config is None and self.wrapper_config_file:
            self.wrapper_config = load_wrapper_config(
                self.wrapper_config_file,
                config_dir=self.config_dir,
                wrapper_kwargs_override=self.wrapper_kwargs,
                default_wrapper_class=self.wrapper_class,
            )

        if self.model_config_file:
            model_path = (
                os.path.join(self.config_dir, self.model_config_file)
                if self.config_dir
                else self.model_config_file
            )
            model_cfg = SB3ModelConfig.from_yaml(model_path)
            if self.algo_config is None:
                self.algo_config = model_cfg.algo_config
            if self.learn_config == SB3LearnConfig():
                self.learn_config = model_cfg.learn_config
            if self.vec_config is None:
                self.vec_config = model_cfg.vec_config
            if self.callback_config is None:
                self.callback_config = model_cfg.callback_config

        model_name_suffix = (
            format_large_number(self.learn_config.total_timesteps)
            if self.learn_config
            else ""
        )
        self.save_config = self.save_config.resolve(
            experiment_id=self.experiment_id,
            model_name_suffix=model_name_suffix,
            replicate_signature=replicate_signature,
        )

        if self.vec_config is not None and not self.vec_config.monitor_dir:
            self.vec_config.monitor_dir = self.save_config.monitor_save_dir

        if self.callback_config is not None:
            eval_cb = self.callback_config.eval_callback_config
            if not eval_cb.best_model_save_path:
                eval_cb.best_model_save_path = self.save_config.eval_save_dir
            if not eval_cb.log_path:
                eval_cb.log_path = self.save_config.eval_save_dir
            if (
                self.callback_config.ckpt_callback_config
                and not self.callback_config.ckpt_callback_config.save_path
            ):
                self.callback_config.ckpt_callback_config.save_path = (
                    self.save_config.model_save_dir
                )
            if (
                self.callback_config.video_recorder_callback_config
                and not self.callback_config.video_recorder_callback_config.save_dir
            ):
                self.callback_config.video_recorder_callback_config.save_dir = (
                    self.save_config.animation_save_path
                )

        return self

    @model_validator(mode="after")
    def _auto_resolve(self) -> Self:
        return self.resolve()


def _format_templates_recursive(
    data: Any,
    replicate_signature: str,
    rep_id: int,
) -> Any:
    """Format template variables recursively in strings, dicts, and lists."""
    if isinstance(data, str):
        try:
            return data.format(
                replicate_signature=replicate_signature, rep_id=rep_id
            )
        except (KeyError, IndexError, ValueError):
            return data
    if isinstance(data, dict):
        return {
            k: _format_templates_recursive(v, replicate_signature, rep_id)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [
            _format_templates_recursive(v, replicate_signature, rep_id)
            for v in data
        ]
    return data


class SB3ReplicatePipelineConfig(BaseModel, YAMLReaderMixin):
    """Configuration for running multiple independent SB3 pipeline replicas."""

    replicate_config: ReplicateConfig
    single_pipeline_config: SB3PipelineConfig | dict[str, Any] | None = None
    ind_pipeline_configs: list[SB3PipelineConfig] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def build_replicates(self) -> Self:
        """Build independent pipeline configurations from single_pipeline_config."""
        if not self.single_pipeline_config or self.ind_pipeline_configs:
            return self

        base_cfg = (
            self.single_pipeline_config
            if isinstance(self.single_pipeline_config, SB3PipelineConfig)
            else SB3PipelineConfig.from_dict(self.single_pipeline_config)
        )

        replicate_pipeline_configs: list[SB3PipelineConfig] = []
        for rep_id in range(self.replicate_config.num_replicates):
            actual_rep_id = rep_id + self.replicate_config.replicate_start_id
            rep_signature = self.replicate_config.replicate_signature.format(
                rep_id=actual_rep_id
            )

            replica = deepcopy(base_cfg)
            replica.resolve(replicate_signature=rep_signature)

            if replica.wrapper_config and replica.wrapper_config.wrapper_kwargs:
                replica.wrapper_config.wrapper_kwargs = (
                    _format_templates_recursive(
                        replica.wrapper_config.wrapper_kwargs,
                        replicate_signature=rep_signature,
                        rep_id=actual_rep_id,
                    )
                )

            base_seed = (
                replica.algo_config.algo_kwargs.get("seed")
                if replica.algo_config
                else None
            )
            if base_seed is None and replica.vec_config:
                base_seed = replica.vec_config.seed
            if base_seed is None:
                base_seed = 10
            rep_seed = int(base_seed) + rep_id * 10
            if replica.algo_config:
                replica.algo_config.algo_kwargs["seed"] = rep_seed
            if replica.vec_config:
                replica.vec_config.seed = rep_seed

            replicate_pipeline_configs.append(replica)

        self.ind_pipeline_configs = replicate_pipeline_configs
        return self

    @model_validator(mode="after")
    def _auto_build(self) -> Self:
        return self.build_replicates()


__all__ = [
    "ArbitraryCallbackConfig",
    "MakeVecEnvConfig",
    "SB3AlgorithmConfig",
    "SB3CallbackConfig",
    "SB3ExperimentManagerConfig",
    "SB3LearnConfig",
    "SB3ModelConfig",
    "SB3OptunaConfig",
    "SB3OptunaDashboardConfig",
    "SB3OptunaParamConfig",
    "SB3PipelineConfig",
    "SB3ReplicatePipelineConfig",
]
