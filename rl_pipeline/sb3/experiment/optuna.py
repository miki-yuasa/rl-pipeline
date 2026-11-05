"""Optuna hyperparameter sampling and range validation utilities."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, TypeGuard

import optuna
from optuna.distributions import CategoricalChoiceType
from stable_baselines3.common.base_class import BaseAlgorithm

if TYPE_CHECKING:
    from rl_pipeline.sb3.config import SB3OptunaParamConfig


def filter_algorithm_kwargs(
    algorithm_class: type[BaseAlgorithm],
    algo_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep only kwargs accepted by the selected algorithm constructor."""
    accepted_params = set(
        inspect.signature(algorithm_class.__init__).parameters
    )
    accepted_params.discard("self")
    return {
        key: value
        for key, value in algo_kwargs.items()
        if key in accepted_params
    }


def sample_params_from_config(
    trial: optuna.trial.BaseTrial,
    tune_params: Sequence[SB3OptunaParamConfig],
) -> dict[str, Any]:
    """Sample hyperparameters from declarative search-space config."""
    sampled: dict[str, Any] = {}

    for param in tune_params:
        target_key = param.target or param.name
        value = _suggest_param(trial=trial, param=param)
        set_nested_value(sampled, target_key, value)

    _validate_sampled_ranges(sampled, trial=trial)
    return sampled


def _check_two_element_range(
    value: Sequence[Any],
    trial: optuna.trial.BaseTrial | None = None,
) -> None:
    """Verify a 2-element range has min < max, pruning the trial otherwise."""
    if len(value) != 2:
        return
    v_min, v_max = value[0], value[1]
    if v_min is None or v_max is None or v_min < v_max:
        return
    if trial is not None:
        trial.set_user_attr("invalid_range", True)
    raise optuna.TrialPruned(
        f"Sampled range min ({v_min}) must be strictly less "
        f"than max ({v_max})."
    )


def _validate_dict_entry(
    key: Any,
    value: Any,
    trial: optuna.trial.BaseTrial | None = None,
) -> None:
    """Validate a single dictionary entry for 2-element ranges."""
    if key == "range" and isinstance(value, (list, tuple)):
        _check_two_element_range(value, trial=trial)
    elif isinstance(value, (Mapping, list)):
        _validate_sampled_ranges(value, trial=trial)


def _validate_sampled_ranges(
    data: Any,
    trial: optuna.trial.BaseTrial | None = None,
) -> None:
    """Validate any 2-element range has min < max; prune trial otherwise."""
    if isinstance(data, Mapping):
        for key, value in data.items():
            _validate_dict_entry(key, value, trial=trial)
        return

    if isinstance(data, list):
        for item in data:
            _validate_sampled_ranges(item, trial=trial)


def _encode_choice(choice: Any) -> CategoricalChoiceType:
    """Normalizes a choice into a primitive type supported by Optuna."""
    if isinstance(choice, (type(None), bool, int, float, str)):
        return choice
    return json.dumps(choice, sort_keys=True)


def _suggest_categorical(
    trial: optuna.trial.BaseTrial,
    param: SB3OptunaParamConfig,
) -> Any:
    """Suggests a categorical value, encoding non-primitives as JSON strings."""
    if not param.choices:
        raise ValueError(
            f"categorical parameter '{param.name}' requires non-empty choices."
        )

    encoded_choices = [_encode_choice(c) for c in param.choices]
    value_by_choice = dict(zip(encoded_choices, param.choices, strict=False))

    chosen = trial.suggest_categorical(param.name, encoded_choices)
    raw_value = value_by_choice[chosen]

    if param.value_mapping is None:
        return raw_value
    return param.value_mapping.get(str(chosen), raw_value)


def decode_trial_params(
    params: Mapping[str, Any],
    tune_params: Sequence[SB3OptunaParamConfig],
) -> dict[str, Any]:
    """Decode Optuna trial parameters back to original search-space types."""
    param_by_name = {p.name: p for p in tune_params}
    decoded: dict[str, Any] = {}

    for name, raw_val in params.items():
        param = param_by_name.get(name)
        if (
            param is None
            or param.suggest_type != "categorical"
            or not param.choices
        ):
            decoded[name] = raw_val
            continue

        value_by_choice = {_encode_choice(c): c for c in param.choices}
        raw_choice = value_by_choice.get(raw_val, raw_val)

        if param.value_mapping is not None:
            decoded[name] = param.value_mapping.get(str(raw_val), raw_choice)
        else:
            decoded[name] = raw_choice

    return decoded


def _require_bounds(
    param: SB3OptunaParamConfig,
    type_label: str,
) -> tuple[float, float]:
    """Return (low, high) as floats or raise ValueError if missing."""
    if param.low is None or param.high is None:
        raise ValueError(
            f"{type_label} parameter '{param.name}' requires low/high"
        )
    return float(param.low), float(param.high)


def _suggest_raw_value(
    trial: optuna.trial.BaseTrial,
    param: SB3OptunaParamConfig,
) -> Any:
    """Suggest raw parameter value from trial based on suggest_type."""
    suggest_type = param.suggest_type
    if suggest_type == "float":
        low, high = _require_bounds(param, "float")
        step = float(param.step) if param.step is not None else None
        return trial.suggest_float(
            param.name,
            low,
            high,
            log=param.log,
            step=step,
        )
    if suggest_type == "int":
        low, high = _require_bounds(param, "int")
        step = int(param.step) if param.step is not None else 1
        return trial.suggest_int(
            param.name,
            int(low),
            int(high),
            step=step,
            log=param.log,
        )
    if suggest_type == "pow2_int":
        low, high = _require_bounds(param, "pow2_int")
        step = int(param.step) if param.step is not None else 1
        exponent = trial.suggest_int(
            param.name, int(low), int(high), step=step
        )
        return 2**exponent
    if suggest_type == "categorical":
        return _suggest_categorical(trial=trial, param=param)
    raise ValueError(f"Unsupported suggest_type: {suggest_type}")


def _suggest_param(
    trial: optuna.trial.BaseTrial,
    param: SB3OptunaParamConfig,
) -> Any:
    value = _suggest_raw_value(trial=trial, param=param)
    if not param.one_minus:
        return value
    if value is None:
        raise ValueError(
            f"Parameter '{param.name}' cannot use one_minus with a None value"
        )
    return 1.0 - float(value)


def _advance_nested_node(
    node: MutableMapping[str, Any],
    key: str,
    next_key: str,
) -> tuple[Any, bool]:
    """Advance one step along a dotted path, returning (next_node, is_range)."""
    if next_key in ("0", "1"):
        if key not in node or not isinstance(node[key], list):
            node[key] = [None, None]
        return node[key], True
    if key not in node or not isinstance(node[key], dict):
        node[key] = {}
    return node[key], False


def set_nested_value(
    target: MutableMapping[str, Any],
    dotted_key: str,
    value: Any,
) -> None:
    """Set a nested dict value, supporting 2-element range indices (.0 and .1).

    Args:
      target: Destination dictionary to mutate.
      dotted_key: Dotted key path, optionally including bracket/index notation.
      value: Value to set at the destination path.
    """
    normalized_key = dotted_key.replace("[", ".").replace("]", "")
    if "." not in normalized_key:
        target[normalized_key] = value
        return

    keys = normalized_key.split(".")
    node: Any = target
    for i, key in enumerate(keys[:-1]):
        node, is_range = _advance_nested_node(node, key, keys[i + 1])
        if is_range:
            break

    if isinstance(node, list) and keys[-1] in ("0", "1"):
        node[int(keys[-1])] = value
    else:
        node[keys[-1]] = value


def _merge_two_element_list(
    existing: Sequence[Any],
    incoming: Sequence[Any],
) -> list[Any]:
    """Merge a 2-element list, keeping existing items when incoming is None."""
    return [
        incoming[0] if incoming[0] is not None else existing[0],
        incoming[1] if incoming[1] is not None else existing[1],
    ]


def _is_two_element_seq(val: Any) -> TypeGuard[list[Any] | tuple[Any, ...]]:
    """Return True if val is a 2-element list or tuple."""
    return isinstance(val, (list, tuple)) and len(val) == 2


def deep_update(
    target: MutableMapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively update nested dicts, supporting partial 2-element ranges."""
    for key, value in source.items():
        existing = target.get(key)
        if isinstance(value, Mapping) and isinstance(existing, MutableMapping):
            deep_update(existing, value)
        elif _is_two_element_seq(value) and _is_two_element_seq(existing):
            target[key] = _merge_two_element_list(existing, value)
        else:
            target[key] = value
    if isinstance(target, dict):
        return target
    return dict(target)


def _assign_sampled_entry(
    key: str,
    value: Any,
    algo_kwargs: MutableMapping[str, Any],
    wrapper_kwargs: MutableMapping[str, Any],
    base_wrapper_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Route a sampled top-level entry into algo_kwargs or wrapper_kwargs."""
    if key in ("wrapper", "wrapper_kwargs") and isinstance(value, Mapping):
        deep_update(wrapper_kwargs, value)
        return
    if key in ("algo", "algo_kwargs") and isinstance(value, Mapping):
        deep_update(algo_kwargs, value)
        return
    if base_wrapper_kwargs and key in base_wrapper_kwargs:
        wrapper_kwargs[key] = value
        return
    algo_kwargs[key] = value


def split_sampled_params(
    sampled_params: Mapping[str, Any],
    base_wrapper_kwargs: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split sampled parameters into algorithm kwargs and wrapper kwargs."""
    algo_kwargs: dict[str, Any] = {}
    wrapper_kwargs: dict[str, Any] = {}

    for key, value in sampled_params.items():
        _assign_sampled_entry(
            key,
            value,
            algo_kwargs=algo_kwargs,
            wrapper_kwargs=wrapper_kwargs,
            base_wrapper_kwargs=base_wrapper_kwargs,
        )

    return algo_kwargs, wrapper_kwargs
