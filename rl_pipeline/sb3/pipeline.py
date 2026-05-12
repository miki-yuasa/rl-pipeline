"""SB3 training and optimization pipelines.

This module contains the single-run SB3 pipeline and replicate pipeline.
It supports standard training/evaluation and Optuna-based hyperparameter
optimization through ``SB3Pipeline.optimize``.
"""

import multiprocessing as mp
import os
import subprocess
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
from gymnasium import Env, Wrapper
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import PolicyPredictor
from stable_baselines3.common.vec_env import VecEnv

from rl_pipeline.core.config import SaveConfig
from rl_pipeline.core.eval.stats import PolicyEvalStats
from rl_pipeline.core.pipeline import BasePipeline
from rl_pipeline.core.utils.io import add_number_to_existing_filepath

from .callback import TrialEvalCallback, VideoRecorderCallback
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
    """Initialize the default callback list for SB3 training.

    Parameters
    ----------
    eval_env : VecEnv
        Evaluation environment used by evaluation callback.
    video_env : Env | Wrapper
        Single environment used by optional video recorder callback.
    callback_config : SB3CallbackConfig
        Callback configuration bundle.

    Returns
    -------
    callbacks : list[BaseCallback]
        Ordered callback list passed to ``model.learn``.
    """
    eval_callback_args = callback_config.eval_callback_config.model_dump()
    # Remove eval_callback_cls from eval_callback_args since it's passed explicitly
    eval_callback_args.pop("eval_callback_cls", None)
    eval_callback = callback_config.eval_callback_config.eval_callback_cls(
        eval_env=eval_env, **eval_callback_args
    )
    callbacks: list[BaseCallback] = [eval_callback]

    if callback_config.ckpt_callback_config:
        ckpt_callback = CheckpointCallback(
            **callback_config.ckpt_callback_config.model_dump()
        )
        callbacks.append(ckpt_callback)

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
    """End-to-end SB3 pipeline for train/evaluate/replay/optimize workflows.

    Parameters
    ----------
    config : SB3PipelineConfig
        Runtime pipeline configuration.
    verbose : bool, optional
        Verbose logging flag, by default ``True``.

    Examples
    --------
    Train and evaluate from an explicit config::

        from stable_baselines3 import PPO

        from rl_pipeline.core.config import SaveConfig
        from rl_pipeline.gymnasium.config import MakeEnvConfig
        from rl_pipeline.sb3 import (
            CheckpointCallbackConfig,
            EvalCallbackConfig,
            MakeVecEnvConfig,
            SB3AlgorithmConfig,
            SB3CallbackConfig,
            SB3LearnConfig,
            SB3Pipeline,
            SB3PipelineConfig,
        )

        config = SB3PipelineConfig(
            device="cuda:0",
            experiment_id="cartpole_explicit",
            save_config=SaveConfig(
                model_save_path="out/models/cartpole/final_model.zip",
                best_model_save_path="out/models/cartpole/best_model.zip",
                monitor_save_dir="out/models/cartpole/monitor",
                tb_save_dir="out/models/cartpole/tb",
                eval_save_dir="out/models/cartpole/eval",
                eval_metrics_save_path="out/models/cartpole/eval/eval_metrics.yaml",
                animation_save_path="out/models/cartpole/animations/final_model.gif",
            ),
            env_config=MakeEnvConfig(id="CartPole-v1", render_mode="rgb_array"),
            vec_config=MakeVecEnvConfig(n_envs=4),
            algo_config=SB3AlgorithmConfig(
                algorithm=PPO,
                algo_kwargs={"policy": "MlpPolicy", "n_steps": 128},
            ),
            learn_config=SB3LearnConfig(total_timesteps=10_000),
            callback_config=SB3CallbackConfig(
                eval_callback_config=EvalCallbackConfig(
                    eval_freq=1_000,
                    n_eval_episodes=5,
                ),
                ckpt_callback_config=CheckpointCallbackConfig(
                    save_freq=1_000,
                    save_path="out/models/cartpole/ckpts",
                ),
            ),
        )
        pipeline = SB3Pipeline(config=config, verbose=True)

        model = pipeline.train()
        eval_result = pipeline.evaluate(checkpoint="best")

    Or load from YAML with a config reader::

        config = SB3PipelineConfigReader.from_yaml("path/to/pipeline.yaml").to_config()

    Run Optuna hyperparameter search::

        from rl_pipeline.sb3 import SB3OptunaConfig

        config.optuna_config = SB3OptunaConfig(
            storage_url="sqlite:///sb3_study.db",
            study_name="cartpole_optuna",
            n_trials=20,
            tune_params=[
                {
                    "name": "learning_rate",
                    "suggest_type": "float",
                    "low": 1e-5,
                    "high": 1e-2,
                    "log": True,
                }
            ],
        )

        pipeline = SB3Pipeline(config=config)
        study = pipeline.optimize()
        print(study.best_trial.value)
    """

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

        Returns
        -------
        model : BaseAlgorithm
            Trained SB3 model instance.
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
        """Build callback list for training.

        Returns
        -------
        callbacks : list[BaseCallback]
            Ordered callback instances.
        """
        return init_callback(
            eval_env=self.env_loader.vec_env(),
            video_env=self.env_loader.env(),
            callback_config=self.callback_configs,
        )

    def _manager_start_run(self):
        """Start experiment manager run if configured."""
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
        """Append experiment-manager callback when available.

        Parameters
        ----------
        callbacks : list[BaseCallback]
            Existing callback list.

        Returns
        -------
        callbacks : list[BaseCallback]
            Callback list with optional manager callback appended.
        """
        if self.experiment_manager and self.config.experiment_manager_config:
            callback = self.experiment_manager.logger_callback(
                self.config.experiment_manager_config.callback_config
            )
            callbacks.append(callback)
        else:
            pass

        return callbacks

    def _manager_end_run(self):
        """End experiment manager run if active."""
        if self.experiment_manager:
            self.experiment_manager.end_run()
        else:
            pass

    def train_on_unsaved_model(self) -> BaseAlgorithm:
        """
        Train the model on an unsaved model.
        This method is a placeholder and should be implemented if needed.

        Returns
        -------
        model : BaseAlgorithm
            Newly trained model or previously saved model loaded from disk.
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
        """Load a model checkpoint or saved model artifact.

        Parameters
        ----------
        ckpt_timestep : int | Literal["latest", "final", "best"], optional
            Which checkpoint/model to load, by default ``"final"``.
        env : Env | Wrapper | None, optional
            Optional environment to bind to the loaded model.
        device : str | None, optional
            Device override for model loading.

        Returns
        -------
        model : BaseAlgorithm
            Loaded SB3 model.
        """
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
                if not self.callback_configs.ckpt_callback_config:
                    raise ValueError(
                        "Checkpoint loading requires a checkpoint callback configuration."
                    )
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
        """Evaluate a model or checkpoint and optionally persist metrics.

        Parameters
        ----------
        n_eval_episodes : int, optional
            Number of evaluation episodes, by default 100.
        deterministic : bool, optional
            Whether to use deterministic actions, by default ``False``.
        save_to_file : bool, optional
            Whether to save evaluation metrics, by default ``True``.
        eval_file_name : str, optional
            Output file name for metrics, by default ``"model_eval.yaml"``.
        checkpoint : int | Literal["latest", "final", "best"] | PolicyPredictor, optional
            Checkpoint identifier or model instance to evaluate.
        env : Literal["single", "vec"] | Env, optional
            Evaluation environment selector or explicit env object.

        Returns
        -------
        eval_result : PolicyEvalStats
            Structured evaluation metrics.
        """

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

        Parameters
        ----------
        model : BaseAlgorithm
            Model to run during replay recording.
        save_path : str | None, optional
            Optional output path; defaults to config animation path.
        custom_player : Callable[[Env, BaseAlgorithm, str, bool], None] | None, optional
            Optional replay recorder implementation.
        verbose : bool, optional
            Verbose flag for replay recorder.
        """
        if save_path is None:
            save_path = self.save_config.animation_save_path

        player = custom_player if custom_player is not None else record_replay
        player(self.env_loader.env(), model, save_path, self.verbose or verbose)

    def optimize(
        self,
        optuna_config: SB3OptunaConfig | None = None,
    ) -> "optuna.Study":
        """Run Optuna hyperparameter optimization for this pipeline.

        Parameters
        ----------
        optuna_config : SB3OptunaConfig | None, optional
            Optional override for optimization configuration. When ``None``,
            ``self.optuna_config`` is used.

        Returns
        -------
        study : optuna.Study
            Optimized Optuna study object.

        Raises
        ------
        ValueError
            If Optuna config or required storage/sampler settings are missing.
        """
        import optuna

        from rl_pipeline.experiment.optuna import (
            build_dashboard_command,
            create_study,
            launch_dashboard,
        )

        from .experiment.optuna import (
            filter_algorithm_kwargs,
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
        storage_url = tune_config.storage_url

        dashboard_command = build_dashboard_command(
            storage_url=storage_url,
            host=tune_config.dashboard.host,
            port=tune_config.dashboard.port,
        )
        if self.verbose:
            print(
                "SB3Pipeline: Optuna dashboard command: "
                f"{dashboard_command.as_shell_command()}"
            )

        study = create_study(
            storage_url=storage_url,
            study_name=tune_config.study_name,
            direction=tune_config.direction,
            n_startup_trials=tune_config.n_startup_trials,
            n_warmup_steps=tune_config.n_warmup_steps,
        )

        if tune_config.dashboard.launch:
            if (
                self.optuna_dashboard_process is None
                or self.optuna_dashboard_process.poll() is not None
            ):
                self.optuna_dashboard_process = launch_dashboard(
                    storage_url=storage_url,
                    host=tune_config.dashboard.host,
                    port=tune_config.dashboard.port,
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
            raise ValueError(
                "Optuna tuning requires either optuna_config.tune_params "
                "or optuna_config.sample_params_fn."
            )

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
            """Objective function executed by Optuna for one trial.

            Parameters
            ----------
            trial : optuna.Trial
                Trial object used for parameter sampling and pruning.

            Returns
            -------
            objective_value : float
                Final objective score (mean reward) for the trial.
            """
            train_env: VecEnv | None = None
            eval_env = None
            model: BaseAlgorithm | None = None
            nan_encountered = False
            trial_artifact_dir = os.path.join(
                self.save_config.model_save_dir,
                "optuna_trials",
                f"trial_{trial.number}",
            )
            os.makedirs(trial_artifact_dir, exist_ok=True)
            best_model_path = os.path.join(trial_artifact_dir, "best_model.zip")
            final_model_path = os.path.join(trial_artifact_dir, "final_model.zip")

            try:
                sampled_algo_kwargs = trial_sampler(trial)
                trial_algo_kwargs = filter_algorithm_kwargs(
                    algorithm_class=algo_class,
                    algo_kwargs={**base_algo_kwargs, **sampled_algo_kwargs},
                )

                train_env = self.env_loader.vec_env()
                eval_env = self.env_loader.vec_env()
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
                    best_model_save_path=trial_artifact_dir,
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

                model.save(os.path.join(trial_artifact_dir, "final_model"))
                trial.set_user_attr("artifact_dir", trial_artifact_dir)
                trial.set_user_attr("best_model_path", best_model_path)
                trial.set_user_attr("final_model_path", final_model_path)

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

        def _optimize_chunk(n_trials_chunk: int) -> None:
            process_study = create_study(
                storage_url=storage_url,
                study_name=tune_config.study_name,
                direction=tune_config.direction,
                n_startup_trials=tune_config.n_startup_trials,
                n_warmup_steps=tune_config.n_warmup_steps,
            )
            process_study.optimize(
                objective,
                n_trials=n_trials_chunk,
                timeout=tune_config.timeout,
                n_jobs=1,
            )

        if tune_config.parallel_backend == "thread":
            study.optimize(
                objective,
                n_trials=tune_config.n_trials,
                timeout=tune_config.timeout,
                n_jobs=tune_config.n_jobs,
            )
            return study

        if tune_config.n_jobs == 1:
            study.optimize(
                objective,
                n_trials=tune_config.n_trials,
                timeout=tune_config.timeout,
                n_jobs=1,
            )
            return study

        if "fork" not in mp.get_all_start_methods():
            if self.verbose:
                print(
                    "SB3Pipeline: Process backend requires the 'fork' start method. "
                    "Falling back to Optuna thread backend."
                )
            study.optimize(
                objective,
                n_trials=tune_config.n_trials,
                timeout=tune_config.timeout,
                n_jobs=tune_config.n_jobs,
            )
            return study

        n_processes = tune_config.n_jobs
        base_trials, remainder = divmod(tune_config.n_trials, n_processes)
        trial_chunks = [
            base_trials + (1 if i < remainder else 0) for i in range(n_processes)
        ]

        ctx = mp.get_context("fork")
        processes: list[Any] = []
        for n_trials_chunk in trial_chunks:
            if n_trials_chunk <= 0:
                continue
            process = ctx.Process(target=_optimize_chunk, args=(n_trials_chunk,))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        failed = [process.pid for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(
                f"Optuna process workers failed with non-zero exit status: {failed}"
            )

        return create_study(
            storage_url=storage_url,
            study_name=tune_config.study_name,
            direction=tune_config.direction,
            n_startup_trials=tune_config.n_startup_trials,
            n_warmup_steps=tune_config.n_warmup_steps,
        )


class SB3ReplicatePipeline:
    """Execute multiple independent SB3 pipelines using replicate configs.

    Parameters
    ----------
    config : SB3ReplicatePipelineConfig
        Replicate pipeline configuration with per-run configs.
    verbose : bool, optional
        Verbose logging flag, by default ``True``.

    Examples
    --------
    Train and evaluate multiple independent runs::

        from rl_pipeline.sb3 import (
            SB3ReplicatePipeline,
            SB3ReplicatePipelineConfigReader,
        )

        rep_config = SB3ReplicatePipelineConfigReader.from_yaml(
            "path/to/replicate_pipeline.yaml"
        ).to_config()
        rep_pipeline = SB3ReplicatePipeline(config=rep_config, verbose=True)

        models = rep_pipeline.train()
        eval_results = rep_pipeline.evaluate(checkpoint="best")
        print(len(models), len(eval_results))
    """

    def __init__(self, config: SB3ReplicatePipelineConfig, verbose: bool = True):
        self.replicate_config = config.replicate_config
        self.ind_pipeline_configs = config.ind_pipeline_configs
        self.ind_pipelines: list[SB3Pipeline] = [
            SB3Pipeline(config=ind_config, verbose=verbose)
            for ind_config in self.ind_pipeline_configs
        ]
        self.verbose = verbose

    def train(self) -> list[BaseAlgorithm]:
        """Train all replicate pipelines.

        Returns
        -------
        models : list[BaseAlgorithm]
            Trained models, one per replicate.
        """
        models: list[BaseAlgorithm] = []
        for ind_pipeline in self.ind_pipelines:
            model = ind_pipeline.train()
            models.append(model)

        return models

    def train_on_unsaved_model(self):
        """Train or load each replicate pipeline model.

        Returns
        -------
        models : list[BaseAlgorithm]
            Models for all replicate pipelines.
        """
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
        """Evaluate all replicate pipelines.

        Returns
        -------
        eval_results : list[PolicyEvalStats]
            Evaluation stats for all replicate runs.
        """
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
        """Load models for all replicate pipelines.

        Returns
        -------
        models : list[BaseAlgorithm]
            Loaded models for each replicate.
        """
        models = []
        for ind_pipeline in self.ind_pipelines:
            model = ind_pipeline.load_model(
                ckpt_timestep=ckpt_timestep, env=env, device=device
            )
            models.append(model)
        return models

    def load_model(
        self,
        rep_idx: int = 0,
        ckpt_timestep: int | Literal["latest", "final", "best"] = "final",
        env: Env | Wrapper | None = None,
        device: str | None = None,
    ) -> list[BaseAlgorithm]:
        """Load models for all replicate pipelines.

        Returns
        -------
        models : list[BaseAlgorithm]
            Loaded models for each replicate.
        """
        model = self.ind_pipelines[rep_idx].load_model(
            ckpt_timestep=ckpt_timestep, env=env, device=device
        )
        return model

    def record_replays(
        self,
        models: list[BaseAlgorithm],
        custom_player: Callable[[Env, BaseAlgorithm, str, bool], None] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Record a replay of the model's performance in the evaluation environment.

        Parameters
        ----------
        models : list[BaseAlgorithm]
            Models corresponding to replicate pipelines.
        custom_player : Callable[[Env, BaseAlgorithm, str, bool], None] | None, optional
            Optional replay recorder implementation.
        verbose : bool, optional
            Verbose flag for replay recorder.
        """
        assert len(models) == len(self.ind_pipelines)

        for model, ind_pipeline in zip(models, self.ind_pipelines):
            ind_pipeline.record_replay(
                model, custom_player=custom_player, verbose=verbose
            )
