from typing import Generic, TypeVar

from pydantic import BaseModel

ExperimentManagerType = TypeVar("ExperimentManagerType")

RunType = TypeVar("RunType")

ManagerConfigType = TypeVar("ManagerConfigType", bound=BaseModel, contravariant=True)


class BaseExperimentManager(Generic[ManagerConfigType, RunType]):
    """
    Base class for experiment managers such as Weights & Biases (wandb) and MLflow.
    """

    def start_run(
        self, manager_config: ManagerConfigType, logged_param_config: BaseModel
    ) -> RunType:
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

    def end_run(self, run: RunType) -> None:
        """
        End the current experiment run.

        Parameters
        ----------
        run : RunType
            The run to be ended.

        """
        raise NotImplementedError("Subclasses must implement end_run method.")
