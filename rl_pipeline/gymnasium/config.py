"""Gymnasium environment and wrapper configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import yaml
from gymnasium import Env, Wrapper
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rl_pipeline.core.config import YAMLReaderMixin
from rl_pipeline.core.utils.instantiate import instantiate
from rl_pipeline.core.utils.io import get_class


@runtime_checkable
class WrapperConvertible(Protocol):
    """Protocol for objects convertible into WrapperConfig."""

    def to_wrapper_config(self) -> WrapperConfig:
        """Convert into a standard WrapperConfig."""
        ...


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

    def wrap(self, env: Env) -> Wrapper:
        """Instantiate and apply the wrapper to a Gymnasium environment.

        Args:
            env: Base environment to wrap.

        Returns:
            Wrapped Gymnasium environment.
        """
        cls = (
            get_class(self.wrapper_class)
            if isinstance(self.wrapper_class, str)
            else self.wrapper_class
        )
        return cls(env, **self.wrapper_kwargs)


class BaseWrapperConfig(BaseModel, YAMLReaderMixin):
    """Base configuration model for strongly-typed environment wrappers."""

    wrapper_class: ClassVar[type[Wrapper] | str | None] = None
    _target_: type[Wrapper] | str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def get_wrapper_class(self) -> type[Wrapper]:
        """Resolve wrapper class from _target_ or wrapper_class class attribute.

        Returns:
            Resolved Wrapper class.

        Raises:
            ValueError: If wrapper class is invalid or unspecified.
        """
        target = self._target_ or self.wrapper_class
        if target is None:
            raise ValueError(
                f"{self.__class__.__name__} requires wrapper_class or _target_."
            )
        if isinstance(target, str):
            resolved = get_class(target)
            if isinstance(resolved, type) and issubclass(resolved, Wrapper):
                return resolved
            raise ValueError(f"{target} must be a subclass of Wrapper")
        return target

    def get_wrapper_kwargs(self) -> dict[str, Any]:
        """Construct dictionary of wrapper keyword arguments.

        Returns:
            Dictionary of recursively instantiated wrapper arguments.
        """
        raw_kwargs = self.model_dump(exclude={"_target_"}, exclude_none=False)
        return instantiate(raw_kwargs)

    def to_wrapper_config(self) -> WrapperConfig:
        """Convert configuration into a standard WrapperConfig.

        Returns:
            Configured WrapperConfig instance.
        """
        return WrapperConfig(
            wrapper_class=self.get_wrapper_class(),
            wrapper_kwargs=self.get_wrapper_kwargs(),
        )

    def to_config(self) -> WrapperConfig:
        """Compatibility alias returning standard WrapperConfig.

        Returns:
            Configured WrapperConfig instance.
        """
        return self.to_wrapper_config()

    def wrap(self, env: Env) -> Wrapper:
        """Instantiate and apply the configured wrapper to an environment.

        Args:
            env: Base environment to wrap.

        Returns:
            Wrapped Gymnasium environment.
        """
        cls = self.get_wrapper_class()
        return cls(env, **self.get_wrapper_kwargs())


class GymEnvConfig(BaseModel, YAMLReaderMixin):
    """Bundled configuration for environment and wrapper creation."""

    make_env_config: MakeEnvConfig
    wrapper_config: WrapperConfig | BaseWrapperConfig | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


def load_wrapper_config(
    data_or_path: (
        str
        | Path
        | Mapping[str, Any]
        | WrapperConfig
        | BaseWrapperConfig
        | None
    ),
    config_dir: str | Path | None = None,
    wrapper_kwargs_override: Mapping[str, Any] | None = None,
    default_wrapper_class: type[Wrapper] | str | None = None,
) -> WrapperConfig | None:
    """Load and resolve a WrapperConfig from a file path, mapping, or model.

    Args:
        data_or_path: File path, mapping, or wrapper configuration model.
        config_dir: Optional base directory if data_or_path is a relative path.
        wrapper_kwargs_override: Optional dict of keyword arguments to overlay.
        default_wrapper_class: Optional fallback wrapper class.

    Returns:
        Instantiated WrapperConfig with resolved class and arguments, or None.

    Raises:
        ValueError: If neither `_target_` nor `wrapper_class` is specified.
    """
    if data_or_path is None:
        return None

    if isinstance(data_or_path, WrapperConvertible):
        cfg = data_or_path.to_wrapper_config()
        if wrapper_kwargs_override:
            cfg.wrapper_kwargs.update(wrapper_kwargs_override)
        return cfg

    if isinstance(data_or_path, WrapperConfig):
        if wrapper_kwargs_override:
            data_or_path.wrapper_kwargs.update(wrapper_kwargs_override)
        return data_or_path

    if isinstance(data_or_path, (str, Path)):
        resolved_path = (
            Path(config_dir) / data_or_path
            if config_dir
            else Path(data_or_path)
        )
        with resolved_path.open("r", encoding="utf-8") as f:
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

    target_cls = (
        get_class(wrapper_target)
        if isinstance(wrapper_target, str)
        else wrapper_target
    )
    if isinstance(target_cls, type):
        config_cls = (
            target_cls
            if issubclass(target_cls, BaseWrapperConfig)
            else getattr(target_cls, "wrapper_config_class", None)
            or getattr(target_cls, "config_class", None)
        )
        if isinstance(config_cls, str):
            config_cls = get_class(config_cls)
        if (
            isinstance(config_cls, type)
            and issubclass(config_cls, BaseWrapperConfig)
        ):
            typed_cfg = config_cls.model_validate(config_by_key)
            res_cfg = typed_cfg.to_wrapper_config()
            if wrapper_kwargs_override:
                res_cfg.wrapper_kwargs.update(wrapper_kwargs_override)
            return res_cfg

    raw_kwargs = (
        config_by_key["wrapper_kwargs"]
        if "wrapper_kwargs" in config_by_key and len(config_by_key) == 1
        else config_by_key
    )
    instantiated_kwargs: dict[str, Any] = instantiate(raw_kwargs)
    if wrapper_kwargs_override:
        instantiated_kwargs.update(wrapper_kwargs_override)

    return WrapperConfig(
        wrapper_class=wrapper_target,
        wrapper_kwargs=instantiated_kwargs,
    )
