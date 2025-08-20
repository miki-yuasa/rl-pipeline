import os
from typing import Any, Generic, Literal, Self, TypeVar

import yaml
from pydantic import BaseModel

from rl_pipeline.core.loader import (
    BaseEnvLoader,
    BaseModelLoader,
    EnvLoaderType,
    ModelLoaderType,
)
from rl_pipeline.eval import PolicyEvalStats
from rl_pipeline.utils.io import get_class

ConfigType = TypeVar("ConfigType", bound=BaseModel)


class BasePipeline(Generic[ConfigType, EnvLoaderType, ModelLoaderType]):
    """
    Base class for training pipelines.
    This class provides a structure for training configurations and methods.
    """

    env_loader: EnvLoaderType
    model_loader: ModelLoaderType

    def __init__(self, config: ConfigType, verbose: bool = True):
        self.config: ConfigType = config
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

    def record_replay(self, model: Any, save_path: str | None = None):
        """
        Placeholder for recording a replay of the model.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """
        Create a pipeline instance from a YAML training configuration file.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        config_class_name: str = data["config_class"]
        config_class: type[ConfigType] | None = get_class(config_class_name)
        assert config_class is not None, f"Config class {config_class_name} not found."
        config: ConfigType = config_class(**data)
        return cls(config)

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

    def load_model(self, ckpt_timestep: int | Literal["latest", "final"]) -> Any:
        raise NotImplementedError(
            "Subclasses should implement the load_model method to load a pre-trained model."
        )
