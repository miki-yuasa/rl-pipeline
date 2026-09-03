import inspect
from typing import TYPE_CHECKING, Any

import optuna
from stable_baselines3.common.base_class import BaseAlgorithm

if TYPE_CHECKING:
    from ..config import SB3OptunaParamConfig


def filter_algorithm_kwargs(
    algorithm_class: type[BaseAlgorithm],
    algo_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Keep only kwargs accepted by the selected algorithm constructor."""
    accepted_params = set(inspect.signature(algorithm_class.__init__).parameters)
    accepted_params.discard("self")
    return {key: value for key, value in algo_kwargs.items() if key in accepted_params}


def sample_params_from_config(
    trial: optuna.trial.BaseTrial,
    tune_params: list["SB3OptunaParamConfig"],
) -> dict[str, Any]:
    """Sample hyperparameters from declarative search-space config."""
    sampled: dict[str, Any] = {}

    for param in tune_params:
        target_key = param.target or param.name
        value = _suggest_param(trial=trial, param=param)
        _set_nested_value(sampled, target_key, value)

    return sampled


def _suggest_param(
    trial: optuna.trial.BaseTrial,
    param: "SB3OptunaParamConfig",
) -> Any:
    suggest_type = param.suggest_type

    if suggest_type == "float":
        if param.low is None or param.high is None:
            raise ValueError(f"float parameter '{param.name}' requires low/high")
        value = trial.suggest_float(
            param.name,
            float(param.low),
            float(param.high),
            log=param.log,
            step=float(param.step) if param.step is not None else None,
        )
    elif suggest_type == "int":
        if param.low is None or param.high is None:
            raise ValueError(f"int parameter '{param.name}' requires low/high")
        value = trial.suggest_int(
            param.name,
            int(param.low),
            int(param.high),
            step=int(param.step) if param.step is not None else 1,
            log=param.log,
        )
    elif suggest_type == "pow2_int":
        if param.low is None or param.high is None:
            raise ValueError(f"pow2_int parameter '{param.name}' requires low/high")
        exponent = trial.suggest_int(
            param.name,
            int(param.low),
            int(param.high),
            step=int(param.step) if param.step is not None else 1,
        )
        value = 2**exponent
    elif suggest_type == "categorical":
        if not param.choices:
            raise ValueError(
                f"categorical parameter '{param.name}' requires non-empty choices"
            )
        choice = trial.suggest_categorical(param.name, param.choices)
        if param.value_mapping is not None:
            value = param.value_mapping.get(str(choice), choice)
        else:
            value = choice
    else:
        raise ValueError(f"Unsupported suggest_type: {suggest_type}")

    if param.one_minus:
        if value is None:
            raise ValueError(
                f"Parameter '{param.name}' cannot use one_minus with a None value"
            )
        value = 1.0 - float(value)

    return value


def _set_nested_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    if "." not in dotted_key:
        target[dotted_key] = value
        return

    keys = dotted_key.split(".")
    node = target
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively update nested dictionaries."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def split_sampled_params(
    sampled_params: dict[str, Any],
    base_wrapper_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split sampled parameters into algorithm kwargs and wrapper kwargs."""
    algo_kwargs: dict[str, Any] = {}
    wrapper_kwargs: dict[str, Any] = {}

    for key, value in sampled_params.items():
        if key in ("wrapper", "wrapper_kwargs") and isinstance(value, dict):
            deep_update(wrapper_kwargs, value)
        elif key in ("algo", "algo_kwargs") and isinstance(value, dict):
            deep_update(algo_kwargs, value)
        elif base_wrapper_kwargs and key in base_wrapper_kwargs:
            wrapper_kwargs[key] = value
        else:
            algo_kwargs[key] = value

    return algo_kwargs, wrapper_kwargs
