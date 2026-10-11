from __future__ import annotations

"""Gymnasium environment and wrapper configuration models."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from gymnasium import Wrapper
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rl_pipeline.core.config import YAMLReaderMixin
from rl_pipeline.core.utils.instantiate import instantiate
from rl_pipeline.core.utils.io import get_class


class MakeEnvConfig(BaseModel, YAMLReaderMixin):
    """Configuration for gymnasium environment creation via gym.make."""

    id: str = "CartPole-v1"
    max_episode_steps: int | None = None
    disable_env_checker: bool | None = None
    render_mode: str = "rgb_array"
    env_kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def make_env_args(self) -> dict[str, Any]:
        """Construct the argument dictionary for gym.make.

        Returns:
            Dictionary of keyword arguments forwarded to gym.make.
        """
        return {
            "id": self.id,
            "max_episode_steps": self.max_episode_steps,
            "disable_env_checker": self.disable_env_checker,
            "render_mode": self.render_mode,
            **self.env_kwargs,
        }


class WrapperConfig(BaseModel, YAMLReaderMixin):
    """Configuration for single-environment wrapper instantiation."""

    wrapper_class: type[Wrapper] | str
    wrapper_kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("wrapper_class", mode="after")
    def resolve_wrapper_class(
        cls, wrapper_class: type[Wrapper] | str
    ) -> type[Wrapper]:
        """Resolve wrapper class from string path if necessary.

        Args:
            wrapper_class: Wrapper class or import path string.

        Returns:
            Resolved wrapper class object.
        """
        if isinstance(wrapper_class, str):
            resolved = get_class(wrapper_class)
            if isinstance(resolved, type) and issubclass(resolved, Wrapper):
                return resolved
            raise ValueError(f"{wrapper_class} must be a subclass of Wrapper")
        return wrapper_class


class GymEnvConfig(BaseModel, YAMLReaderMixin):
    """Bundled configuration for environment and wrapper creation."""

    make_env_config: MakeEnvConfig
    wrapper_config: WrapperConfig | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


def load_wrapper_config(
    data_or_path: str | Path | Mapping[str, Any] | None,
    config_dir: str | Path | None = None,
    wrapper_kwargs_override: Mapping[str, Any] | None = None,
    default_wrapper_class: type[Wrapper] | str | None = None,
) -> WrapperConfig | None:
    """Load and resolve a WrapperConfig from a file path or dictionary.

    Args:
        data_or_path: File path or mapping containing wrapper configuration.
        config_dir: Optional base directory if data_or_path is a relative path.
        wrapper_kwargs_override: Optional dictionary of keyword arguments to overlay.
        default_wrapper_class: Optional fallback wrapper class if none is specified.

    Returns:
        Instantiated WrapperConfig with resolved class and instantiated arguments,
        or None if data_or_path is None.

    Raises:
        ValueError: If neither `_target_` nor `wrapper_class` is specified.
    """
    if data_or_path is None:
        return None

    if isinstance(data_or_path, (str, Path)):
        resolved_path = (
            os.path.join(config_dir, str(data_or_path))
            if config_dir
            else str(data_or_path)
        )
        with open(resolved_path, "r", encoding="utf-8") as f:
            config_by_key = yaml.safe_load(f) or {}
    else:
        config_by_key = dict(data_or_path)

    wrapper_target = (
        config_by_key.pop("_target_", None)
        or config_by_key.pop("wrapper_class", None)
        or default_wrapper_class
    )
    if wrapper_target is None:
        raise ValueError(
            "Wrapper configuration must specify `_target_` or `wrapper_class`."
        )

    instantiated_kwargs: dict[str, Any] = instantiate(config_by_key)
    if wrapper_kwargs_override:
        instantiated_kwargs.update(wrapper_kwargs_override)

    return WrapperConfig(
        wrapper_class=wrapper_target,
        wrapper_kwargs=instantiated_kwargs,
    )
