from __future__ import annotations

"""Declarative object instantiation mirroring Meta Hydra's instantiate pattern.

This module implements recursive object instantiation from declarative mapping
specifications containing `_target_`, matching the behavior of
`hydra.utils.instantiate` from Meta's Hydra configuration framework.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from rl_pipeline.core.utils.io import get_class


def _instantiate_mapping(
    mapping_data: Mapping[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Helper to instantiate a mapping specification.

    Args:
        mapping_data: Mapping potentially containing `_target_`.
        *args: Positional arguments forwarded to target callable.
        **kwargs: Keyword arguments forwarded to target callable.

    Returns:
        Instantiated object if `_target_` is present, or a new dictionary with
        recursively instantiated values.
    """
    if "_target_" in mapping_data:
        target_name = mapping_data["_target_"]
        target_callable = (
            get_class(target_name)
            if isinstance(target_name, str)
            else target_name
        )
        kwargs_by_name = {
            k: instantiate(v)
            for k, v in mapping_data.items()
            if k != "_target_"
        }
        kwargs_by_name.update(kwargs)
        return target_callable(*args, **kwargs_by_name)

    return {k: instantiate(v) for k, v in mapping_data.items()}


def _instantiate_sequence(sequence_data: Sequence[Any]) -> list[Any]:
    """Helper to instantiate elements of a sequence.

    Args:
        sequence_data: Sequence of configuration elements.

    Returns:
        List of instantiated elements.
    """
    return [instantiate(item) for item in sequence_data]


def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
    """Recursively instantiate objects from configurations with `_target_`.

    Mirrors the behavior of `hydra.utils.instantiate` from Meta's Hydra.

    Args:
        config: Configuration mapping, sequence, or primitive value.
        *args: Positional arguments forwarded to top-level target callable.
        **kwargs: Keyword arguments forwarded to top-level target callable.

    Returns:
        Instantiated object or configuration with nested targets instantiated.
    """
    if isinstance(config, Mapping):
        return _instantiate_mapping(config, *args, **kwargs)
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        return _instantiate_sequence(config)
    return config
