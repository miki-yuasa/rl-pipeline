"""Pre-validation ingestion and dictionary resolution for SB3 pipelines."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from rl_pipeline.gymnasium.config import MakeEnvConfig, load_wrapper_config


def _resolve_config_dir(
    data_by_key: Mapping[str, Any],
    base_dir: Path | str | None = None,
) -> str | None:
    """Extract configuration dir from mapping or fallback to base directory.

    Args:
        data_by_key: Raw configuration mapping to extract from.
        base_dir: Fallback directory if not specified in mapping.

    Returns:
        Resolved base directory string, or None.
    """
    config_dir = data_by_key.get("config_dir")
    if config_dir is not None:
        warnings.warn(
            "`config_dir` is deprecated and will be removed in a future"
            " release. Please specify file paths relative to your workspace"
            " root or as absolute paths.",
            FutureWarning,
            stacklevel=2,
        )
        return str(config_dir)
    if base_dir is not None:
        return str(base_dir)
    return None


def _resolve_env_config(
    data_by_key: MutableMapping[str, Any],
    config_dir: str | None,
) -> None:
    """Resolve and inline environment configuration from file reference.

    Args:
        data_by_key: Raw configuration mapping to modify in-place.
        config_dir: Base directory for relative file paths.
    """
    if "env_config" not in data_by_key and "env_config_file" in data_by_key:
        env_file = data_by_key.pop("env_config_file")
        env_path = (
            str(Path(config_dir) / env_file) if config_dir else str(env_file)
        )
        data_by_key["env_config"] = MakeEnvConfig.from_yaml(env_path)
    else:
        data_by_key.pop("env_config_file", None)


def _resolve_model_config(
    data_by_key: MutableMapping[str, Any],
    config_dir: str | None,
) -> None:
    """Resolve algorithm and learning configs from model file reference.

    Args:
        data_by_key: Raw configuration mapping to modify in-place.
        config_dir: Base directory for relative file paths.
    """
    if "model_config_file" not in data_by_key:
        return

    model_file = data_by_key.pop("model_config_file")
    model_path = (
        str(Path(config_dir) / model_file) if config_dir else str(model_file)
    )

    from rl_pipeline.sb3.config import SB3ModelConfig

    model_cfg = SB3ModelConfig.from_yaml(model_path)
    data_by_key.setdefault("algo_config", model_cfg.algo_config)
    data_by_key.setdefault("learn_config", model_cfg.learn_config)
    data_by_key.setdefault("vec_config", model_cfg.vec_config)
    data_by_key.setdefault("callback_config", model_cfg.callback_config)


def _resolve_wrapper_config(
    data_by_key: MutableMapping[str, Any],
    config_dir: str | None,
) -> None:
    """Resolve and inline environment wrapper configuration.

    Args:
        data_by_key: Raw configuration mapping to modify in-place.
        config_dir: Base directory for relative file paths.
    """
    wrapper_file = data_by_key.pop("wrapper_config_file", None)
    inline_wrapper = data_by_key.get("wrapper_config")
    wrapper_cls = data_by_key.pop("wrapper_class", None)
    wrapper_kwargs = data_by_key.pop("wrapper_kwargs", None)

    if wrapper_file or inline_wrapper or wrapper_cls:
        data_by_key["wrapper_config"] = load_wrapper_config(
            inline_wrapper or wrapper_file,
            config_dir=config_dir,
            wrapper_kwargs_override=wrapper_kwargs,
            default_wrapper_class=wrapper_cls,
        )


def ingest_pipeline_config_dict(
    data_by_key: Mapping[str, Any],
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Resolves and inlines external file references before Pydantic validation.

    Args:
        data_by_key: Raw configuration dictionary from YAML or user input.
        base_dir: Optional base directory to resolve relative file paths.

    Returns:
        Clean dictionary with resolved configuration objects and stripped keys.
    """
    resolved_by_key = dict(data_by_key)
    config_dir = _resolve_config_dir(resolved_by_key, base_dir)

    _resolve_env_config(resolved_by_key, config_dir)
    _resolve_model_config(resolved_by_key, config_dir)
    _resolve_wrapper_config(resolved_by_key, config_dir)
    resolved_by_key.pop("config_dir", None)

    return resolved_by_key
