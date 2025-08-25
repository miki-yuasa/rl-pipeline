from typing import Generic, TypeVar

from stable_baselines3.common.callbacks import BaseCallback

from rl_pipeline.core.experiment import RunType

SB3ExperimentManagerType = TypeVar(
    "SB3ExperimentManagerType", bound="SB3ExperimentManager"
)


class SB3ExperimentManager(Generic[RunType]):
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
