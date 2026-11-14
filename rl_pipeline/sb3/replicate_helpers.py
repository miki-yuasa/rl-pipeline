"""Helper utilities for resolving replicate pipeline templates and seeds."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from rl_pipeline.core.utils.io import get_class
from rl_pipeline.core.utils.seed import sample_base_seed

if TYPE_CHECKING:
    from rl_pipeline.core import ReplicateConfig

    from .config import SB3PipelineConfig


def format_templates_recursive(
    data: Any,
    replicate_signature: str,
    rep_id: int,
) -> Any:
    """Format template variables recursively in strings, dicts, and lists."""
    if isinstance(data, str):
        try:
            return data.format(
                replicate_signature=replicate_signature, rep_id=rep_id
            )
        except (KeyError, IndexError, ValueError):
            return data
    if isinstance(data, dict):
        return {
            k: format_templates_recursive(v, replicate_signature, rep_id)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [
            format_templates_recursive(v, replicate_signature, rep_id)
            for v in data
        ]
    return data


def resolve_pipeline_cls(
    pipeline_by_key: Mapping[str, Any],
    base_cls: type[SB3PipelineConfig],
) -> type[SB3PipelineConfig]:
    """Resolve custom pipeline config class from _target_ if present."""
    if "_target_" in pipeline_by_key:
        raw_target = pipeline_by_key["_target_"]
        target = (
            get_class(raw_target) if isinstance(raw_target, str) else raw_target
        )
        if isinstance(target, type) and issubclass(target, base_cls):
            return target
    return base_cls


def build_replica_from_source(
    source_config: Mapping[str, Any] | SB3PipelineConfig,
    replicate_signature: str,
    rep_id: int,
    base_cls: type[SB3PipelineConfig],
) -> SB3PipelineConfig:
    """Instantiate and resolve a single replica from source config."""
    if isinstance(source_config, base_cls):
        replica = deepcopy(source_config)
        replica.resolve(replicate_signature=replicate_signature)
        wrap_cfg = replica.wrapper_config
        if wrap_cfg and wrap_cfg.wrapper_kwargs:
            wrap_cfg.wrapper_kwargs = format_templates_recursive(
                wrap_cfg.wrapper_kwargs,
                replicate_signature=replicate_signature,
                rep_id=rep_id,
            )
        return replica

    replica_by_key = format_templates_recursive(
        deepcopy(dict(source_config)),
        replicate_signature=replicate_signature,
        rep_id=rep_id,
    )
    pipeline_cls = resolve_pipeline_cls(replica_by_key, base_cls)
    replica_by_key.pop("_target_", None)
    replica = pipeline_cls.from_dict(replica_by_key)
    replica.resolve(replicate_signature=replicate_signature)
    return replica


def extract_explicit_seed(replica: SB3PipelineConfig) -> int | None:
    """Extract an explicit seed from algorithm or vectorized env config."""
    seed: int | None = (
        replica.algo_config.algo_kwargs.get("seed")
        if replica.algo_config
        else None
    )
    if seed is None and replica.vec_config:
        seed = replica.vec_config.seed
    if seed is None:
        return None
    resolved = int(seed)
    if resolved < 0:
        raise ValueError(f"Seed must be non-negative, got {resolved}.")
    return resolved


def resolve_base_seed(
    replicate_cfg: ReplicateConfig,
    sample_replica: SB3PipelineConfig,
) -> int:
    """Resolve explicit base seed or sample and record a random root seed."""
    if replicate_cfg.base_seed is not None:
        return replicate_cfg.base_seed
    explicit_seed = extract_explicit_seed(sample_replica)
    if explicit_seed is not None:
        return explicit_seed
    sampled_seed = sample_base_seed()
    replicate_cfg.base_seed = sampled_seed
    return sampled_seed
