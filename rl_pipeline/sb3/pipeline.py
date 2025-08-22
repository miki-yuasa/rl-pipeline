import os
from typing import Literal

import numpy as np
from gymnasium import Env, Wrapper
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from rl_pipeline.core.config import SaveConfig
from rl_pipeline.core.eval.stats import PolicyEvalStats
from rl_pipeline.core.pipeline import BasePipeline
from rl_pipeline.core.utils.io import add_number_to_existing_filepath

from .callback import SuccessEvalCallback, VideoRecorderCallback
from .config import SB3CallbackConfig, SB3LearnConfig, SB3PipelineConfig
from .loader import SB3EnvLoader, SB3ModelLoader
from .utils import SuccessBuffer, SuccessBufferEval, record_replay


def init_callback(
    eval_env: VecEnv, video_env: Env | Wrapper, callback_config: SB3CallbackConfig
) -> list[BaseCallback]:
    ckpt_callback = CheckpointCallback(
        **callback_config.ckpt_callback_config.model_dump()
    )
    eval_callback = SuccessEvalCallback(
        eval_env=eval_env, **callback_config.eval_callback_config.model_dump()
    )
    callbacks = [ckpt_callback, eval_callback]

    if callback_config.video_recorder_callback_config:
        video_callback = VideoRecorderCallback(
            eval_env=video_env,
            **callback_config.video_recorder_callback_config.model_dump(),
        )
        callbacks.append(video_callback)

    return callbacks


class SB3Pipeline(BasePipeline[SB3PipelineConfig, SB3EnvLoader, SB3ModelLoader]):
    def __init__(self, config: SB3PipelineConfig, verbose: bool = True):
        super().__init__(config, verbose=verbose)

        self.save_config: SaveConfig = config.save_config
        self.learn_config: SB3LearnConfig = config.learn_config
        self.callback_configs: SB3CallbackConfig = config.callback_config

        self.env_loader = SB3EnvLoader(
            config.env_config, config.wrapper_config, config.vec_config
        )
        self.model_loader = SB3ModelLoader(
            config.algo_config, config.save_config.tb_save_dir
        )

    def train(self) -> BaseAlgorithm:
        """
        Train the model using the provided training configuration.
        """
        train_env: VecEnv = self.env_loader.vec_env()
        model: BaseAlgorithm = self.model_loader.model(
            train_env, device=self.config.device
        )
        callback: list[BaseCallback] = self._init_callback()

        model.learn(**self.learn_config.model_dump(), callback=callback)
        train_env.close()
        os.makedirs(self.save_config.model_save_dir, exist_ok=True)
        # Save the model
        # if there is already an existing model, add a number suffix e.g. "_1"
        save_path: str = add_number_to_existing_filepath(
            self.save_config.model_save_path
        )
        model.save(save_path)

        return model

    def _init_callback(self) -> list[BaseCallback]:
        return init_callback(
            eval_env=self.env_loader.vec_env(),
            video_env=self.env_loader.env(),
            callback_config=self.callback_configs,
        )

    def train_on_unsaved_model(self) -> BaseAlgorithm:
        """
        Train the model on an unsaved model.
        This method is a placeholder and should be implemented if needed.
        """
        demo_env = self.env_loader.env()
        if (
            not os.path.exists(self.config.save_config.model_save_path)
            or self.config.retrain_model
        ):
            model = self.train()
        else:
            print(
                f"SB3Pipeline: Model {self.config.save_config.model_save_path} already exists, loading..."
            )
            model = self.model_loader.load_model(
                self.config.save_config.model_save_path, demo_env, self.config.device
            )
        return model

    def load_model(
        self,
        ckpt_timestep: int | Literal["latest", "final", "best"] = "final",
        env: Env | Wrapper | None = None,
        device: str | None = None,
    ) -> BaseAlgorithm:
        if device is None:
            device = self.config.device

        match ckpt_timestep:
            case "final":
                model = self.model_loader.load_model(
                    self.config.save_config.model_save_path, env, device
                )

            case "best":
                model = self.model_loader.load_model(
                    self.config.save_config.best_model_save_path, env, device
                )

            case _:
                model = self.model_loader.load_checkpoint(
                    ckpt_dir=self.callback_configs.ckpt_callback_config.save_path,
                    ckpt_name_prefix=self.callback_configs.ckpt_callback_config.name_prefix,
                    timestep=ckpt_timestep,
                    env=env,
                    device=device,
                    file_ext=".zip",
                )

        return model

    def evaluate(
        self,
        n_eval_episodes: int = 100,
        deterministic: bool = False,
        save_to_file: bool = True,
        eval_file_name: str = "model_eval.yaml",
        checkpoint: int | Literal["latest", "final", "best"] | BaseAlgorithm = "final",
    ) -> PolicyEvalStats:
        """Evaluate the final model."""

        if self.verbose:
            print(f"SB3Pipeline: Evaluating the {checkpoint} model...")

        if isinstance(checkpoint, BaseAlgorithm):
            model: BaseAlgorithm = checkpoint
        else:
            model: BaseAlgorithm = self.load_model(checkpoint)

        success_buffer = SuccessBuffer()
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            self.env_loader.vec_env(),
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
            callback=success_buffer._log_success_callback,
        )
        assert isinstance(episode_rewards, list)
        assert isinstance(episode_lengths, list)

        # Only save four decimal places for readability
        decimal_places: int = 4
        mean_reward: float = float(np.mean(episode_rewards).round(decimal_places))
        std_reward: float = float(np.std(episode_rewards).round(decimal_places))
        mean_episode_length: float = float(
            np.mean(episode_lengths).round(decimal_places)
        )
        std_episode_length: float = float(np.std(episode_lengths).round(decimal_places))
        success_buffer_result: SuccessBufferEval = success_buffer.post_eval()

        eval_result = PolicyEvalStats(
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_episode_length=mean_episode_length,
            std_episode_length=std_episode_length,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            **success_buffer_result.model_dump(),
        )

        if self.verbose:
            self._print_eval_result(eval_result)

        if save_to_file:
            eval_file_path = os.path.join(
                self.save_config.eval_save_dir, eval_file_name
            )
            modified_eval_file_path = add_number_to_existing_filepath(eval_file_path)
            self._save_eval_result(eval_result, modified_eval_file_path)

        return eval_result

    def record_replay(
        self, model: BaseAlgorithm, save_path: str | None = None, verbose: bool = True
    ) -> None:
        """
        Record a replay of the model's performance in the evaluation environment.
        """
        if save_path is None:
            save_path = self.save_config.animation_save_path

        record_replay(self.env_loader.env(), model, save_path, self.verbose or verbose)
