from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from .config import ConfigReader, YAMLReaderMixin
from .utils.io import get_class

ExperimentManagerType = TypeVar("ExperimentManagerType")

RunType = TypeVar("RunType")


class BaseExperimentManager(Generic[RunType]):
    """
    Base class for experiment managers such as Weights & Biases (wandb) and MLflow.
    """

    def __init__(self) -> None:
        self.run: RunType | None = None

    def start_run(self, manager_config, logged_param_config: BaseModel) -> RunType:
        """
        Start a new experiment run.

        Parameters
        ----------
        manager_config : ConfigType
            Configuration for the experiment manager.
        logged_param_config : ConfigType
            Configuration for the logged parameters.

        Returns
        -------
        run: RunType
            The started run for the current experiment.

        """
        raise NotImplementedError("Subclasses must implement start_run method.")

    def end_run(self) -> None:
        """
        End the current experiment run.

        Parameters
        ----------
        run : RunType
            The run to be ended.

        """
        raise NotImplementedError("Subclasses must implement end_run method.")


class ExperimentManagerConfig(BaseModel):
    manager_class: type[BaseExperimentManager]
    manager_config: dict[str, Any]


class ExperimentManagerConfigReader(
    BaseModel, ConfigReader[ExperimentManagerConfig], YAMLReaderMixin
):
    manager_class: str
    manager_config: dict[str, Any]

    def to_config(self) -> ExperimentManagerConfig:
        manager_class = get_class(self.manager_class)
        assert manager_class is not None, (
            f"Could not find experiment manager class for {self.manager_class}"
        )
        return ExperimentManagerConfig(
            manager_class=manager_class, manager_config=self.manager_config
        )
