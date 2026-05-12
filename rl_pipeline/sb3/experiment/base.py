from stable_baselines3.common.callbacks import BaseCallback

from rl_pipeline.core.experiment import (
    BaseExperimentManager,
    CallbackConfigType,
    LoggedParamConfigType,
    ManagerConfigType,
    PipelineConfigType,
    RunType,
)


class SB3ExperimentManager(
    BaseExperimentManager[
        PipelineConfigType,
        RunType,
        ManagerConfigType,
        LoggedParamConfigType,
        CallbackConfigType,
    ]
):
    """
    Protocol for experiment managers in the SB3 framework.

    This protocol defines the methods that must be implemented by any
    experiment manager used within the SB3 framework, assuming the logging
    feature is enabled through a callback (e.g., WandbCallback) derived from
    the BaseCallback class.
    """

    def __init__(self, config: PipelineConfigType) -> None:
        # Initialize WandbExperimentManager
        ...

    def start_run(
        self,
        manager_config: ManagerConfigType,
        logged_param_config: LoggedParamConfigType,
    ) -> RunType: ...

    def logger_callback(self, callback_config: CallbackConfigType) -> BaseCallback:
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

    @staticmethod
    def add_run_name_suffix(
        manager_config: ManagerConfigType,
        run_name_suffix: str,
    ) -> ManagerConfigType:
        """
        Add a suffix to the run name in the manager config.
        Default behavior is just passing.

        Parameters
        ----------
        manager_config
            The manager configuration dictionary.
        run_name_suffix : str
            The suffix to add to the run name.
        """

        return manager_config
