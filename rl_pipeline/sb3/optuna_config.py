from __future__ import annotations

"""Optuna hyperparameter tuning configuration models."""

from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from rl_pipeline.core.config import YAMLReaderMixin
from rl_pipeline.core.utils.io import get_class


class SB3OptunaDashboardConfig(BaseModel, YAMLReaderMixin):
    """Configuration for optuna-dashboard visualization server."""

    launch: bool = False
    host: str | None = None
    port: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="allow")


class SB3OptunaParamConfig(BaseModel, YAMLReaderMixin):
    """Specification for a single tunable parameter search space."""

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

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("value_mapping", mode="after")
    def _resolve_value_mapping(
        cls, value_mapping: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        if value_mapping is None:
            return None
        resolved: dict[str, Any] = {}
        for key, value in value_mapping.items():
            if isinstance(value, str) and "." in value:
                try:
                    resolved[key] = get_class(value)
                except (AttributeError, ImportError, ModuleNotFoundError):
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved


class SB3OptunaConfig(BaseModel, YAMLReaderMixin):
    """Configuration for Optuna hyperparameter optimization."""

    storage_url: str | None = None
    study_name: str | None = None
    direction: Literal["maximize", "minimize"] = "maximize"
    metric: str = "mean_reward"
    n_trials: int = Field(ge=1, default=50)
    timeout: int | None = Field(default=None, ge=1)
    n_jobs: int = Field(ge=1, default=1)
    parallel_backend: Literal["process", "thread"] = "process"
    n_startup_trials: int = Field(ge=0, default=5)
    n_warmup_steps: int = Field(ge=0, default=0)
    n_evaluations: int = Field(ge=1, default=2)
    n_eval_episodes: int = Field(ge=1, default=3)
    deterministic_eval: bool = True
    total_timesteps: int | None = Field(default=None, ge=1)
    eval_freq: int | None = Field(default=None, ge=1)
    sample_params_fn: Callable[[Any], dict[str, Any]] | str | None = None
    tune_params: list[SB3OptunaParamConfig] = Field(default_factory=list)
    dashboard: SB3OptunaDashboardConfig = SB3OptunaDashboardConfig()

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("sample_params_fn", mode="after")
    def _resolve_sample_params_fn(
        cls, fn: Callable[[Any], dict[str, Any]] | str | None
    ) -> Callable[[Any], dict[str, Any]] | None:
        if isinstance(fn, str):
            resolved = get_class(fn)
            return resolved
        return fn

    @field_serializer("sample_params_fn", when_used="json")
    def serialize_sample_params_fn(
        self, sample_params_fn: Callable[[Any], dict[str, Any]] | None
    ) -> str | None:
        """Serialize custom sampling callable for JSON output."""
        if sample_params_fn is None:
            return None
        fn_name = getattr(
            sample_params_fn, "__name__", sample_params_fn.__class__.__name__
        )
        fn_mod = getattr(
            sample_params_fn,
            "__module__",
            sample_params_fn.__class__.__module__,
        )
        return f"{fn_mod}.{fn_name}"
