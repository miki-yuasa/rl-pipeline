from __future__ import annotations

"""Algorithm and learning configuration models for SB3."""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.core.config import YAMLReaderMixin
from rl_pipeline.core.utils.io import get_class


def class_to_string(cls: type) -> str:
    """Serialize a Python class to a fully qualified import path."""
    return f"{cls.__module__}.{cls.__name__}"


def _resolve_with_prefix(algo: str, prefix: str) -> type[BaseAlgorithm] | None:
    """Attempt resolving an algorithm with a specific module prefix."""
    try:
        candidate = get_class(f"{prefix}{algo}")
        if isinstance(candidate, type) and issubclass(candidate, BaseAlgorithm):
            return candidate
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None
    return None


def resolve_algorithm(algo: type[BaseAlgorithm] | str) -> type[BaseAlgorithm]:
    """Resolve an algorithm name or path into an SB3-compatible class."""
    if isinstance(algo, type) and issubclass(algo, BaseAlgorithm):
        return algo
    if isinstance(algo, str):
        if "." in algo:
            cls = get_class(algo)
            if isinstance(cls, type) and issubclass(cls, BaseAlgorithm):
                return cls
        for prefix in ("stable_baselines3.", "sb3_contrib."):
            candidate = _resolve_with_prefix(algo, prefix)
            if candidate is not None:
                return candidate
    raise ValueError(f"Could not resolve algorithm class: {algo}")


class SB3AlgorithmConfig(BaseModel, YAMLReaderMixin):
    """Algorithm class and constructor kwargs for SB3 model creation.

    Follows Hydra's instantiate pattern: supports declarative specification
    via ``_target_`` or ``algorithm`` with constructor kwargs.
    """

    algorithm: type[BaseAlgorithm] | None = None
    algo_kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def _normalize_algo_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        dict_by_name = dict(data)
        raw_algo = dict_by_name.pop("_target_", None) or dict_by_name.pop(
            "algorithm", "PPO"
        )
        algo_cls = resolve_algorithm(raw_algo)
        kwargs_by_name = dict_by_name.pop("algo_kwargs", {})
        if not isinstance(kwargs_by_name, dict):
            kwargs_by_name = {}
        kwargs_by_name.update(dict_by_name)
        return {"algorithm": algo_cls, "algo_kwargs": kwargs_by_name}

    @field_serializer("algorithm", when_used="json")
    def serialize_algorithm(self, algorithm: type[BaseAlgorithm]) -> str:
        return class_to_string(algorithm)


class SB3LearnConfig(BaseModel, YAMLReaderMixin):
    """Training-loop arguments forwarded to ``model.learn``."""

    total_timesteps: int = Field(ge=1, default=1_000_000)
    log_interval: int = Field(ge=0, default=100)
    tb_log_name: str = "run"
    reset_num_timesteps: bool = True
    progress_bar: bool = False

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
