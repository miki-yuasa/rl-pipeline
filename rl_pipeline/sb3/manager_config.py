"""Configuration models for experiment manager integration."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from rl_pipeline.core.config import YAMLReaderMixin
from rl_pipeline.core.utils.io import get_class

from .algo_config import class_to_string
from .experiment.base import SB3ExperimentManager


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
