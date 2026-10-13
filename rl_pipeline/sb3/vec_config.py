"""Configuration models for SB3 vectorized environments."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_pipeline.core.config import YAMLReaderMixin
from rl_pipeline.core.utils.io import get_class

from .algo_config import class_to_string


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
