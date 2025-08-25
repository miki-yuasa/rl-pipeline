from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel
from stable_baselines3.common.callbacks import BaseCallback

from rl_pipeline.core.config import ConfigReader, YAMLReaderMixin
from rl_pipeline.core.experiment import RunType
from rl_pipeline.core.utils.io import get_class

SB3ExperimentManagerType = TypeVar(
    "SB3ExperimentManagerType", bound="SB3ExperimentManager"
)


class SB3ExperimentManager(Protocol, Generic[RunType]):
    """
    Protocol for experiment managers in the SB3 framework.

    This protocol defines the methods that must be implemented by any
    experiment manager used within the SB3 framework, assuming the logging
    feature is enabled through a callback (e.g., WandbCallback) derived from
    the BaseCallback class.
    """

    def start_run(self, manager_config, logged_param_config) -> RunType: ...

    def logger_callback(self, callback_config) -> BaseCallback:
        """
        Create a logger callback for the experiment manager.

        Parameters
        ----------
        callback_config : BaseModel
            Configuration for the logger callback.

        Returns
        -------
        callback: CallbackType
            The logger callback for the experiment manager.
        """
        ...

    def end_run(self) -> None: ...


class SB3ExperimentManagerConfig(BaseModel):
    manager_class: type[SB3ExperimentManager]
    manager_config: dict[str, Any]
    callback_config: dict[str, Any]


class SB3ExperimentManagerConfigReader(
    BaseModel, ConfigReader[SB3ExperimentManagerConfig], YAMLReaderMixin
):
    manager_class: str
    manager_config: dict[str, Any]
    callback_config: dict[str, Any]

    def to_config(self) -> SB3ExperimentManagerConfig:
        manager_class = get_class(self.manager_class)
        assert manager_class is not None, (
            f"Could not find experiment manager class for {self.manager_class}"
        )
        return SB3ExperimentManagerConfig(
            manager_class=manager_class,
            manager_config=self.manager_config,
            callback_config=self.callback_config,
        )
