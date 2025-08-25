from typing import Any, Literal

from pydantic import BaseModel, Field
from wandb.integration.sb3 import WandbCallback
from wandb.sdk.wandb_run import Run

from rl_pipeline.experiment.wandb import WandbExperimentManager, WandbInitConfig

from ..config import SB3PipelineConfig
from .base import SB3ExperimentManager


class WandbCallbackConfig(BaseModel):
    """
    Configuration for the WandbCallback.

    Attributes
    ----------
    verbose: int = 0
        The verbosity of sb3 output
    gradient_save_freq: int = 0
        Frequency to log gradient. The default value is 0 so the gradients are not logged
    log: Literal["gradients", "parameters", "all"] | None = "all"
        What to log. One of "gradients", "parameters", or "all".
    """

    verbose: int = Field(ge=0, default=0)
    gradient_save_freq: int = Field(ge=0, default=0)
    log: Literal["gradients", "parameters", "all"] | None = "all"


class SB3WandbExperimentManager(
    WandbExperimentManager[SB3PipelineConfig], SB3ExperimentManager[Run]
):
    def __init__(self, config: SB3PipelineConfig) -> None:
        super().__init__(config)

    def start_run(
        self,
        manager_config: dict[str, Any] | WandbInitConfig,
        logged_param_config: BaseModel,
    ) -> Run:
        return super().start_run(manager_config, logged_param_config)

    def end_run(self) -> None:
        return super().end_run()

    def logger_callback(self, callback_config: dict[str, Any] | WandbCallbackConfig):
        callback_config_dict = (
            callback_config
            if isinstance(callback_config, dict)
            else callback_config.model_dump()
        )
        return WandbCallback(
            **callback_config_dict,
        )
