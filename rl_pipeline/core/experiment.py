from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

from .config import YAMLReaderMixin
from .typing import PipelineConfigType
from .utils.io import get_class

ExperimentManagerType = TypeVar("ExperimentManagerType")

RunType = TypeVar("RunType")
ManagerConfigType = TypeVar("ManagerConfigType")
LoggedParamConfigType = TypeVar("LoggedParamConfigType")
CallbackConfigType = TypeVar("CallbackConfigType")


class BaseExperimentManager(
    Generic[
        PipelineConfigType,
        RunType,
        ManagerConfigType,
        LoggedParamConfigType,
    ]
):
    """Base class for experiment managers such as Weights & Biases (wandb) and MLflow."""

    def __init__(self, config: PipelineConfigType) -> None:
        self.config: PipelineConfigType = config
        self.run: RunType | None = None

    def start_run(
        self,
        manager_config: ManagerConfigType,
        logged_param_config: LoggedParamConfigType,
    ) -> RunType:
        """Start a new experiment run."""
        raise NotImplementedError("Subclasses must implement start_run method.")

    def end_run(self) -> None:
        """End the current experiment run."""
        raise NotImplementedError("Subclasses must implement end_run method.")


class ExperimentManagerConfig(BaseModel, YAMLReaderMixin):
    """Configuration for an experiment manager."""

    manager_class: type[BaseExperimentManager] | str
    manager_config: dict[str, Any]

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("manager_class", mode="after")
    @classmethod
    def _resolve_manager_class(cls, val: Any) -> type[BaseExperimentManager]:
        if isinstance(val, str):
            resolved = get_class(val)
            return resolved
        return val
