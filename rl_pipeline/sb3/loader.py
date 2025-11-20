from typing import Any, Generic, Literal

import gymnasium as gym
from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from rl_pipeline.core.loader import BaseEnvLoader, BaseModelLoader
from rl_pipeline.core.utils.io import get_ckpt_file, get_file_with_largest_number
from rl_pipeline.gymnasium.config import MakeEnvConfig, WrapperConfig

from .config import MakeVecEnvConfig, SB3AlgorithmConfig
from .utils.env import make_vec_env


class SB3EnvLoader(
    BaseEnvLoader, Generic[WrapperObsType, WrapperActType, ObsType, ActType]
):
    def __init__(
        self,
        env_config: MakeEnvConfig,
        wrapper_config: WrapperConfig | None,
        vec_config: MakeVecEnvConfig | None,
    ) -> None:
        super().__init__()
        self.env_config: MakeEnvConfig = env_config
        self.wrapper_config: WrapperConfig | None = wrapper_config
        self.vec_config: MakeVecEnvConfig | None = vec_config

    def env(
        self,
    ) -> (
        Env[ObsType, ActType]
        | Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]
    ):
        env: Env[ObsType, ActType] = gym.make(**self.env_config.make_env_args())
        if self.wrapper_config:
            wrapped_env: Wrapper[WrapperObsType, WrapperActType, ObsType, ActType] = (
                self.wrapper_config.wrapper_class(
                    env, **self.wrapper_config.wrapper_kwargs
                )
            )
            return wrapped_env
        else:
            return env

    def vec_env(self) -> VecEnv:
        assert self.vec_config is not None, "vec_config must be provided"

        wrapper_class = None
        wrapper_kwargs = None
        if self.wrapper_config:
            wrapper_class = self.wrapper_config.wrapper_class
            wrapper_kwargs = self.wrapper_config.wrapper_kwargs

        env_kwargs: dict[str, Any] = self.env_config.env_kwargs | {
            "max_episode_steps": self.env_config.max_episode_steps,
            "disable_env_checker": self.env_config.disable_env_checker,
            "render_mode": self.env_config.render_mode,
        }

        return make_vec_env(
            env_id=self.env_config.id,
            n_envs=self.vec_config.n_envs,
            seed=self.vec_config.seed,
            start_index=self.vec_config.start_index,
            monitor_dir=self.vec_config.monitor_dir,
            env_kwargs=env_kwargs,
            wrapper_class=wrapper_class,
            wrapper_kwargs=wrapper_kwargs,
            vec_env_cls=self.vec_config.vec_env_cls,
            vec_env_kwargs=self.vec_config.vec_env_kwargs,
            monitor_kwargs=self.vec_config.monitor_kwargs,
        )


class SB3ModelLoader(
    BaseModelLoader, Generic[WrapperObsType, WrapperActType, ObsType, ActType]
):
    def __init__(self, algo_config: SB3AlgorithmConfig, tb_save_dir: str):
        super().__init__()
        self.algo_config: SB3AlgorithmConfig = algo_config
        self.tb_save_dir: str = tb_save_dir

    def model(
        self,
        env: Env[ObsType, ActType]
        | Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]
        | VecEnv,
        device: str,
    ) -> BaseAlgorithm:
        """Return the model."""
        return self.algo_config.algorithm(
            **self.algo_config.algo_kwargs,
            env=env,
            tensorboard_log=self.tb_save_dir,
            device=device,
        )

    def load_model(
        self,
        filepath: str,
        env: Env[ObsType, ActType]
        | Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]
        | None = None,
        device: str = "cuda:0",
    ) -> BaseAlgorithm:
        """Load the model from the checkpoint."""

        # if there is a file with number extension (e.g. _1.zip), pick the one with the largest number
        modified_filepath: str = get_file_with_largest_number(filepath)

        return self.algo_config.algorithm.load(
            modified_filepath, env=env, device=device
        )

    def load_checkpoint(
        self,
        ckpt_dir: str,
        ckpt_name_prefix: str,
        timestep: int | Literal["latest"],
        env: Env[ObsType, ActType]
        | Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]
        | None = None,
        device: str = "cuda:0",
        file_ext: str = ".zip",
    ) -> BaseAlgorithm:
        """Load the model from the checkpoint."""
        ckpt_filepath: str = get_ckpt_file(
            ckpt_dir, ckpt_name_prefix, timestep, file_ext
        )
        return self.load_model(ckpt_filepath, env=env, device=device)
