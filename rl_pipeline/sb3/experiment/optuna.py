import inspect
from typing import TYPE_CHECKING, Any

import optuna
import torch.nn as nn
from stable_baselines3.common.base_class import BaseAlgorithm

if TYPE_CHECKING:
    from ..config import SB3OptunaParamConfig


def sample_params(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """Default hyperparameter sampler for SB3 algorithms."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)

    exponent_n_steps = trial.suggest_int("exponent_n_steps", 3, 10)
    n_steps = 2**exponent_n_steps

    exponent_batch_size = trial.suggest_int("exponent_batch_size", 3, 10)
    batch_size = min(2**exponent_batch_size, n_steps)

    net_arch_key = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    activation_key = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])

    net_arch_map: dict[str, dict[str, list[int]]] = {
        "tiny": {"pi": [64], "vf": [64]},
        "small": {"pi": [64, 64], "vf": [64, 64]},
        "medium": {"pi": [256, 256], "vf": [256, 256]},
    }
    activation_map: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
    }

    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch_map[net_arch_key],
            "activation_fn": activation_map[activation_key],
            "ortho_init": ortho_init,
        },
    }


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
