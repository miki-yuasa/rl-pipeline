import os
import subprocess
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
from gymnasium import Env, Wrapper
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import PolicyPredictor
from stable_baselines3.common.vec_env import VecEnv

from rl_pipeline.core.config import SaveConfig
from rl_pipeline.core.eval.stats import PolicyEvalStats
from rl_pipeline.core.pipeline import BasePipeline
from rl_pipeline.core.utils.io import add_number_to_existing_filepath

from .callback import SuccessEvalCallback, TrialEvalCallback, VideoRecorderCallback
from .config import (
    SB3CallbackConfig,
    SB3LearnConfig,
    SB3OptunaConfig,
    SB3PipelineConfig,
    SB3ReplicatePipelineConfig,
)
from .experiment import SB3ExperimentManager
from .loader import SB3EnvLoader, SB3ModelLoader
from .utils import SuccessBuffer, SuccessBufferEval, record_replay

if TYPE_CHECKING:
    import optuna


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

    for arbitrary_callback_config in callback_config.arbitrary_callback_configs:
        callback_class = arbitrary_callback_config.callback_class
        callback_instance = callback_class(**arbitrary_callback_config.callback_kwargs)
        callbacks.append(callback_instance)

    return callbacks


class SB3Pipeline(
    BasePipeline[SB3PipelineConfig, SB3EnvLoader, SB3ModelLoader, SB3ExperimentManager],
):
    def __init__(self, config: SB3PipelineConfig, verbose: bool = True):
        super().__init__(config, verbose=verbose)

        self.save_config: SaveConfig = config.save_config
        self.learn_config: SB3LearnConfig = config.learn_config
        self.callback_configs: SB3CallbackConfig = config.callback_config
        self.optuna_config: SB3OptunaConfig | None = config.optuna_config
        self.optuna_dashboard_process: subprocess.Popen[str] | None = None

        self.env_loader = SB3EnvLoader(
            config.env_config, config.wrapper_config, config.vec_config
        )
        self.model_loader = SB3ModelLoader(
            config.algo_config, config.save_config.tb_save_dir
        )

        if self.config.experiment_manager_config:
            manager_class = self.config.experiment_manager_config.manager_class
            self.experiment_manager = manager_class(config=config)

    def train(self) -> BaseAlgorithm:
        """
        Train the model using the provided training configuration.
        """

        self._manager_start_run()

        train_env: VecEnv = self.env_loader.vec_env()
        model: BaseAlgorithm = self.model_loader.model(
            train_env, device=self.config.device
        )
        callback: list[BaseCallback] = self._init_callback()
        callback = self._manager_add_callback(callback)

        model.learn(**self.learn_config.model_dump(), callback=callback)
        train_env.close()
        os.makedirs(self.save_config.model_save_dir, exist_ok=True)
        # Save the model
        # if there is already an existing model, add a number suffix e.g. "_1"
        save_path: str = add_number_to_existing_filepath(
            self.save_config.model_save_path
        )
        model.save(save_path)

        self._manager_end_run()

        return model

    def _init_callback(self) -> list[BaseCallback]:
        return init_callback(
            eval_env=self.env_loader.vec_env(),
            video_env=self.env_loader.env(),
            callback_config=self.callback_configs,
        )

    def _manager_start_run(self):
        if self.experiment_manager and self.config.experiment_manager_config:
            manager_config: dict[str, Any] = (
                self.config.experiment_manager_config.manager_config
            )
            manager_config.update(
                {
                    "id": self.unique_id(),
                    "name": manager_config["name"] + f"_{self.exp_time()}"
                    if manager_config.get("name")
                    else f"run_{self.exp_time()}",
                }
            )
            self.experiment_manager.start_run(
                manager_config=manager_config, logged_param_config=self.config
            )
        else:
            pass

    def _manager_add_callback(self, callbacks: list[BaseCallback]):
        if self.experiment_manager and self.config.experiment_manager_config:
            callback = self.experiment_manager.logger_callback(
                self.config.experiment_manager_config.callback_config
            )
            callbacks.append(callback)
        else:
            pass

        return callbacks

    def _manager_end_run(self):
        if self.experiment_manager:
            self.experiment_manager.end_run()
        else:
            pass

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
        checkpoint: int
        | Literal["latest", "final", "best"]
        | PolicyPredictor = "final",
        env: Literal["single", "vec"] | Env = "vec",
    ) -> PolicyEvalStats:
        """Evaluate the final model."""

        if self.verbose:
            print(f"SB3Pipeline: Evaluating the {checkpoint} model...")

        model: PolicyPredictor
        if isinstance(checkpoint, int) or isinstance(checkpoint, str):
            model = self.load_model(ckpt_timestep=checkpoint)
        else:
            model = checkpoint

        match env:
            case "single":
                eval_env = self.env_loader.env()
            case "vec":
                eval_env = self.env_loader.vec_env()
            case Env():
                eval_env = env
            case _:
                raise ValueError("Invalid env type for evaluation.")

        success_buffer = SuccessBuffer()
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
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
        self,
        model: BaseAlgorithm,
        save_path: str | None = None,
        custom_player: Callable[[Env, BaseAlgorithm, str, bool], None] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Record a replay of the model's performance in the evaluation environment.
        """
        if save_path is None:
            save_path = self.save_config.animation_save_path

        player = custom_player if custom_player is not None else record_replay
        player(self.env_loader.env(), model, save_path, self.verbose or verbose)

    def optimize(
        self,
        optuna_config: SB3OptunaConfig | None = None,
    ) -> "optuna.Study":
        import optuna

        from rl_pipeline.experiment.optuna import (
            build_dashboard_command,
            create_study,
            launch_dashboard,
        )

        from .experiment.optuna import (
            filter_algorithm_kwargs,
            sample_params,
            sample_params_from_config,
        )

        tune_config = optuna_config if optuna_config is not None else self.optuna_config
        if tune_config is None:
            raise ValueError(
                "optuna_config is required. Set SB3PipelineConfig.optuna_config or pass it to optimize()."
            )

        if not tune_config.storage_url:
            raise ValueError(
                "optuna_config.storage_url is required for persistent studies and optuna-dashboard."
            )

        if tune_config.dashboard.launch:
            if (
                self.optuna_dashboard_process is None
                or self.optuna_dashboard_process.poll() is not None
            ):
                self.optuna_dashboard_process = launch_dashboard(
                    storage_url=tune_config.storage_url,
                    host=tune_config.dashboard.host,
                    port=tune_config.dashboard.port,
                )

        dashboard_command = build_dashboard_command(
            storage_url=tune_config.storage_url,
            host=tune_config.dashboard.host,
            port=tune_config.dashboard.port,
        )
        if self.verbose:
            print(
                "SB3Pipeline: Optuna dashboard command: "
                f"{dashboard_command.as_shell_command()}"
            )

        study = create_study(
            storage_url=tune_config.storage_url,
            study_name=tune_config.study_name,
            direction=tune_config.direction,
            n_startup_trials=tune_config.n_startup_trials,
            n_warmup_steps=tune_config.n_warmup_steps,
        )

        if tune_config.tune_params:

            def _trial_sampler_from_config(
                trial: optuna.trial.BaseTrial,
            ) -> dict[str, Any]:
                return sample_params_from_config(trial, tune_config.tune_params)

            trial_sampler = _trial_sampler_from_config
        elif tune_config.sample_params_fn is not None:
            sample_params_fn = tune_config.sample_params_fn
            assert sample_params_fn is not None

            def _trial_sampler_from_callable(
                trial: optuna.trial.BaseTrial,
            ) -> dict[str, Any]:
                return sample_params_fn(trial)

            trial_sampler = _trial_sampler_from_callable

        else:
            trial_sampler = sample_params

        algo_class = self.config.algo_config.algorithm
        base_algo_kwargs = deepcopy(self.config.algo_config.algo_kwargs)

        total_timesteps = (
            tune_config.total_timesteps
            if tune_config.total_timesteps is not None
            else self.learn_config.total_timesteps
        )
        n_envs = self.config.vec_config.n_envs if self.config.vec_config else 1
        eval_freq = max(total_timesteps // tune_config.n_evaluations // n_envs, 1)

        def objective(trial: optuna.Trial) -> float:
            train_env: VecEnv | None = None
            eval_env = None
            model: BaseAlgorithm | None = None
            nan_encountered = False

            try:
                sampled_algo_kwargs = trial_sampler(trial)
                trial_algo_kwargs = filter_algorithm_kwargs(
                    algorithm_class=algo_class,
                    algo_kwargs={**base_algo_kwargs, **sampled_algo_kwargs},
                )

                train_env = self.env_loader.vec_env()
                eval_env = Monitor(self.env_loader.env())
                model = algo_class(
                    **trial_algo_kwargs,
                    env=train_env,
                    tensorboard_log=self.save_config.tb_save_dir,
                    device=self.config.device,
                )

                eval_callback = TrialEvalCallback(
                    eval_env=eval_env,
                    trial=trial,
                    n_eval_episodes=tune_config.n_eval_episodes,
                    eval_freq=eval_freq,
                    deterministic=tune_config.deterministic_eval,
                    verbose=0,
                )

                learn_kwargs = self.learn_config.model_dump()
                learn_kwargs["total_timesteps"] = total_timesteps
                learn_kwargs["tb_log_name"] = (
                    f"{self.learn_config.tb_log_name}_trial_{trial.number}"
                )

                try:
                    model.learn(**learn_kwargs, callback=eval_callback)
                except AssertionError as e:
                    if self.verbose:
                        print(
                            "SB3Pipeline: AssertionError during Optuna trial "
                            f"{trial.number}: {e}"
                        )
                    nan_encountered = True

                if nan_encountered:
                    return float("nan")

                if eval_callback.is_pruned:
                    raise optuna.exceptions.TrialPruned()

                if eval_callback.last_mean_reward is None:
                    return float("-inf")
                return float(eval_callback.last_mean_reward)

            finally:
                if model is not None and model.env is not None:
                    model.env.close()
                if train_env is not None:
                    train_env.close()
                if eval_env is not None:
                    eval_env.close()

        study.optimize(
            objective,
            n_trials=tune_config.n_trials,
            timeout=tune_config.timeout,
            n_jobs=tune_config.n_jobs,
        )
        return study


class SB3ReplicatePipeline:
    def __init__(self, config: SB3ReplicatePipelineConfig, verbose: bool = True):
        self.replicate_config = config.replicate_config
        self.ind_pipeline_configs = config.ind_pipeline_configs
        self.ind_pipelines: list[SB3Pipeline] = [
            SB3Pipeline(config=ind_config, verbose=verbose)
            for ind_config in self.ind_pipeline_configs
        ]
        self.verbose = verbose

    def train(self) -> list[BaseAlgorithm]:
        models: list[BaseAlgorithm] = []
        for ind_pipeline in self.ind_pipelines:
            model = ind_pipeline.train()
            models.append(model)

        return models

    def train_on_unsaved_model(self):
        models: list[BaseAlgorithm] = []
        for ind_pipeline in self.ind_pipelines:
            model = ind_pipeline.train_on_unsaved_model()
            models.append(model)

        return models

    def evaluate(
        self,
        n_eval_episodes: int = 100,
        deterministic: bool = False,
        save_to_file: bool = True,
        eval_file_name: str = "model_eval.yaml",
        checkpoint: int | Literal["latest", "final", "best"] | BaseAlgorithm = "final",
    ) -> list[PolicyEvalStats]:
        eval_results = []
        for ind_pipeline in self.ind_pipelines:
            result = ind_pipeline.evaluate(
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic,
                save_to_file=save_to_file,
                eval_file_name=eval_file_name,
                checkpoint=checkpoint,
            )
            eval_results.extend(result)
        return eval_results

    def load_models(
        self,
        ckpt_timestep: int | Literal["latest", "final", "best"] = "final",
        env: Env | Wrapper | None = None,
        device: str | None = None,
    ) -> list[BaseAlgorithm]:
        models = []
        for ind_pipeline in self.ind_pipelines:
            model = ind_pipeline.load_model(
                ckpt_timestep=ckpt_timestep, env=env, device=device
            )
            models.append(model)
        return models

    def record_replays(
        self,
        models: list[BaseAlgorithm],
        custom_player: Callable[[Env, BaseAlgorithm, str, bool], None] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Record a replay of the model's performance in the evaluation environment.
        """
        assert len(models) == len(self.ind_pipelines)

        for model, ind_pipeline in zip(models, self.ind_pipelines):
            ind_pipeline.record_replay(
                model, custom_player=custom_player, verbose=verbose
            )
