from __future__ import annotations

"""Callback configuration models for SB3 pipelines."""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from rl_pipeline.core.config import YAMLReaderMixin
from rl_pipeline.core.utils.io import get_class

from .algo_config import class_to_string


class EvalCallbackConfig(BaseModel, YAMLReaderMixin):
    """Configuration for the Stable Baselines3 evaluation callback."""

    eval_callback_cls: type[EvalCallback] | str = Field(
        default="rl_pipeline.sb3.callback.SuccessEvalCallback",
        validate_default=True,
    )
    eval_freq: int = Field(ge=0, description="Frequency of evaluation in timesteps.")
    n_eval_episodes: int = Field(ge=1, description="Number of episodes to evaluate.")
    log_path: str = Field(default="eval", description="Evaluation logs directory.")
    best_model_save_path: str | None = Field(default=None)
    deterministic: bool = Field(default=False)
    render: bool = Field(default=False)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("eval_callback_cls", mode="after")
    def _resolve_callback_cls(
        cls, callback_cls: type[EvalCallback] | str
    ) -> type[EvalCallback]:
        if isinstance(callback_cls, str):
            resolved = get_class(callback_cls)
            if isinstance(resolved, type) and issubclass(resolved, EvalCallback):
                return resolved
            raise ValueError(f"{callback_cls} must be an EvalCallback subclass")
        return callback_cls


class CheckpointCallbackConfig(BaseModel, YAMLReaderMixin):
    """Configuration for the Stable Baselines3 checkpoint callback."""

    save_freq: int = Field(ge=1, description="Frequency of saving the model.")
    save_path: str = Field(default="ckpts", description="Checkpoint directory.")
    name_prefix: str = Field(default="ckpt")
    save_replay_buffer: bool = Field(default=False)
    verbose: int = Field(default=0)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class VideoRecorderCallbackConfig(BaseModel, YAMLReaderMixin):
    """Configuration for the Stable Baselines3 video recorder callback."""

    render_freq: int = Field(ge=1, default=100)
    save_dir: str = "out/animation"
    name_prefix: str = "rl_model"
    file_ext: str = "gif"
    deterministic: bool = False

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class ArbitraryCallbackConfig(BaseModel, YAMLReaderMixin):
    """Configuration for attaching user-defined SB3 callbacks."""

    callback_class: type[BaseCallback] | str
    callback_kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("callback_class", mode="after")
    def _resolve_callback_class(cls, val: Any) -> type[BaseCallback]:
        if isinstance(val, str):
            resolved = get_class(val)
            if isinstance(resolved, type) and issubclass(resolved, BaseCallback):
                return resolved
            raise ValueError(f"{val} must be a BaseCallback subclass")
        return val

    @field_serializer("callback_class", when_used="json")
    def serialize_callback_class(self, callback_class: type[BaseCallback]) -> str:
        return class_to_string(callback_class)


class SB3CallbackConfig(BaseModel, YAMLReaderMixin):
    """Grouped callback settings used by the SB3 pipeline."""

    eval_callback_config: EvalCallbackConfig
    ckpt_callback_config: CheckpointCallbackConfig | None = None
    video_recorder_callback_config: VideoRecorderCallbackConfig | None = None
    arbitrary_callback_configs: list[ArbitraryCallbackConfig] = Field(
        default_factory=list
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
