"""Configuration models for SB3 training, eval, and replicate pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Self

import yaml
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
from .ingestion import ingest_pipeline_config_dict
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
                return get_class(val)
        return val

    @field_serializer("vec_env_cls", when_used="json")
    def serialize_vec_env_cls(
        self, vec_env_cls: type[SubprocVecEnv | DummyVecEnv] | None
    ) -> str | None:
        """Serialize vector environment class into class string."""
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
            return get_class(val)
        return val

    @field_serializer("manager_class", when_used="json")
    def serialize_manager_class(
        self, manager_class: type[SB3ExperimentManager]
    ) -> str:
        """Serialize manager class into class string."""
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

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def load_ingested_data(cls, data: Any) -> Any:
        """Ingest external configuration references before model validation."""
        if isinstance(data, Mapping):
            return ingest_pipeline_config_dict(data)
        return data

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> Self:
        """Load configuration from a YAML file.

        Args:
            filepath: Path to the YAML file.

        Returns:
            Instantiated and resolved SB3PipelineConfig.
        """
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @field_validator("device", mode="before")
    def _normalize_device(cls, val: Any) -> str:
        if isinstance(val, int):
            return f"cuda:{val}"
        return str(val)

    def resolve(self, replicate_signature: str = "") -> Self:
        """Propagate artifact paths across save, vec, and callbacks."""
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
            vrec_cb = self.callback_config.video_recorder_callback_config
            if vrec_cb and not vrec_cb.save_dir:
                vrec_cb.save_dir = self.save_config.animation_save_path

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

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> Self:
        """Load replicate configuration from a YAML file.

        Args:
            filepath: Path to the YAML file.

        Returns:
            Instantiated SB3ReplicatePipelineConfig with resolved replicas.
        """
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def build_replicates(self) -> Self:
        """Build independent configs from single_pipeline_config."""
        if not self.single_pipeline_config or self.ind_pipeline_configs:
            return self

        replicate_pipeline_configs: list[SB3PipelineConfig] = []
        for rep_id in range(self.replicate_config.num_replicates):
            actual_rep_id = rep_id + self.replicate_config.replicate_start_id
            rep_signature = self.replicate_config.replicate_signature.format(
                rep_id=actual_rep_id
            )

            if isinstance(self.single_pipeline_config, dict):
                replica_dict = _format_templates_recursive(
                    deepcopy(self.single_pipeline_config),
                    replicate_signature=rep_signature,
                    rep_id=actual_rep_id,
                )
                replica = SB3PipelineConfig.from_dict(replica_dict)
                replica.resolve(replicate_signature=rep_signature)
            else:
                is_pipeline_cfg = isinstance(
                    self.single_pipeline_config, SB3PipelineConfig
                )
                base_cfg = (
                    self.single_pipeline_config
                    if is_pipeline_cfg
                    else SB3PipelineConfig.from_dict(
                        self.single_pipeline_config
                    )
                )
                replica = deepcopy(base_cfg)
                replica.resolve(replicate_signature=rep_signature)
                wrap_cfg = replica.wrapper_config
                if wrap_cfg and wrap_cfg.wrapper_kwargs:
                    wrap_cfg.wrapper_kwargs = (
                        _format_templates_recursive(
                            wrap_cfg.wrapper_kwargs,
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
