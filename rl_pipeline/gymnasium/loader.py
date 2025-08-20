from typing import Generic

import gymnasium as gym
from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from rl_pipeline.core.loader import BaseEnvLoader

from .config import GymEnvConfig


class GymEnvLoader(
    BaseEnvLoader, Generic[WrapperObsType, WrapperActType, ObsType, ActType]
):
    def __init__(self, config: GymEnvConfig):
        self.config = config

    def env(
        self,
    ) -> (
        Env[ObsType, ActType]
        | Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]
    ):
        env: Env[ObsType, ActType] = gym.make(
            **self.config.make_env_config.make_env_args()
        )
        if self.config.wrapper_config:
            wrapped_env: Wrapper[WrapperObsType, WrapperActType, ObsType, ActType] = (
                self.config.wrapper_config.wrapper_class(
                    env, **self.config.wrapper_config.wrapper_args
                )
            )
            return wrapped_env
        else:
            return env
