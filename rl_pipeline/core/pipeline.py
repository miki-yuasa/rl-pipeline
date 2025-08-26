import os
import uuid
from datetime import datetime
from typing import Any, Generic, Literal

import yaml

from .eval import PolicyEvalStats
from .experiment import ExperimentManagerType
from .loader import EnvLoaderType, ModelLoaderType
from .typing import PipelineConfigType


class BasePipeline(
    Generic[PipelineConfigType, EnvLoaderType, ModelLoaderType, ExperimentManagerType]
):
    """
    Base class for training pipelines.
    This class provides a structure for training configurations and methods.
    """

    env_loader: EnvLoaderType
    model_loader: ModelLoaderType
    experiment_manager: ExperimentManagerType | None = None

    def __init__(self, config: PipelineConfigType, verbose: bool = True):
        self.config: PipelineConfigType = config
        self.verbose: bool = verbose

    def train(self) -> Any:
        """
        Placeholder for the training method.
        Should be implemented in subclasses.

        Returns
        ---------
        model: The trained model.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def train_on_unsaved_model(self):
        """
        Placeholder for training on an unsaved model.
        Should be implemented in subclasses.

        Returns
        ---------
        model: The trained model.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate(self):
        """
        Placeholder for the evaluation method.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def load_model(
        self, ckpt_timestep: int | Literal["latest", "final", "best"]
    ) -> Any:
        raise NotImplementedError(
            "Subclasses should implement the load_model method to load a pre-trained model."
        )

    def record_replay(
        self,
        model: Any,
        save_path: str | None = None,
        custom_player=None,
        verbose: bool = True,
    ):
        """
        Placeholder for recording a replay of the model.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _print_eval_result(self, eval_result: PolicyEvalStats):
        """
        Print evaluation results in a formatted way.
        """
        if self.verbose:
            print("- Evaluation results:")
            print(
                f" - Reward: Mean {eval_result.mean_reward}, STD {eval_result.std_reward}"
            )
            print(
                f" - Episode length: Mean {eval_result.mean_episode_length}, STD {eval_result.std_episode_length}"
            )
            if eval_result.success_rate is not None:
                print(f" - Success rate: {eval_result.success_rate:.2f}")
            if eval_result.failure_rate is not None:
                print(f" - Failure rate: {eval_result.failure_rate:.2f}")

    def _save_eval_result(self, eval_result: PolicyEvalStats, eval_file_path: str):
        """
        Save evaluation results to a file.
        """
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, "w") as f:
            yaml.dump(eval_result.model_dump(), f, sort_keys=False)
        if self.verbose:
            print(f"Saved evaluation results to {eval_file_path}")

    def unique_id(self) -> str:
        """Generate a unique identifier for the training run."""
        return str(uuid.uuid4())[:4]

    def exp_time(self) -> str:
        """Get the current time formatted for the experiment."""
        return datetime.now().strftime("%m-%d_%H-%M-%S.%f")
