from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Self
from unittest import mock

import gymnasium as gym
import optuna
from absl.testing import absltest, parameterized
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from torch import nn

from rl_pipeline.core import ReplicateConfig
from rl_pipeline.experiment.optuna import build_dashboard_command
from rl_pipeline.gymnasium.config import WrapperConfig
from rl_pipeline.sb3 import (
    SB3OptunaConfig,
    SB3OptunaDashboardConfig,
    SB3Pipeline,
    SB3PipelineConfigReader,
)
from rl_pipeline.sb3.callback import TrialEvalCallback, TrialLossCallback
from rl_pipeline.sb3.config import SB3OptunaParamConfig, SB3PipelineConfig
from rl_pipeline.sb3.config_reader import (
    SB3PipelineConfigReader as SB3PipelineConfigReaderClass,
)
from rl_pipeline.sb3.config_reader import (
    SB3ReplicatePipelineConfigReader,
)
from rl_pipeline.sb3.experiment.optuna import (
    decode_trial_params,
    deep_update,
    filter_algorithm_kwargs,
    sample_params_from_config,
    split_sampled_params,
)


class DummyAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        env: Any = None,
        tensorboard_log: Any = None,
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        self.env = env
        self.device = device or "cpu"

    def learn(self, *args: Any, callback: Any = None, **kwargs: Any) -> Self:
        if callback is not None and hasattr(callback, "last_mean_reward"):
            callback.last_mean_reward = 1.0
        return self

    def save(
        self,
        path: str | Path | Any,
        exclude: Any = None,
        include: Any = None,
    ) -> None:
        out_path = Path(path if str(path).endswith(".zip") else f"{path}.zip")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("dummy")

    def _setup_model(self) -> None:
        pass


class SB3OptunaTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir = Path(self.create_tempdir().full_path)

    def _create_pipeline_config(self) -> SB3PipelineConfig:
        config = SB3PipelineConfigReader.from_yaml(
            "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
        ).to_config()
        self.assertIsNotNone(config.vec_config)
        assert config.vec_config is not None
        config.vec_config.vec_env_cls = DummyVecEnv
        config.vec_config.n_envs = 1
        config.algo_config.algorithm = DummyAlgorithm
        config.save_config.model_save_path = str(
            self.temp_dir / "models" / "final.zip"
        )
        config.save_config.best_model_save_path = str(
            self.temp_dir / "models" / "best.zip"
        )
        config.experiment_manager_config = None
        return config

    def test_filter_algorithm_kwargs_drops_unknown_values(self) -> None:
        kwargs = {"learning_rate": 1e-3, "gamma": 0.99, "unknown_param": 123}
        filtered = filter_algorithm_kwargs(PPO, kwargs)
        self.assertIn("learning_rate", filtered)
        self.assertIn("gamma", filtered)
        self.assertNotIn("unknown_param", filtered)

    def test_dashboard_command_builder(self) -> None:
        command = build_dashboard_command(
            storage_url="sqlite:///study.db",
            host="0.0.0.0",
            port=8080,
        )
        self.assertEqual(
            command.command,
            [
                "optuna-dashboard",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "sqlite:///study.db",
            ],
        )

    def test_sample_params_from_config_supports_nested_and_mapping(
        self,
    ) -> None:
        trial = optuna.trial.FixedTrial(
            {"exp_n_steps": 5, "gamma_eps": 0.01, "activation": "relu"}
        )
        tune_params = [
            SB3OptunaParamConfig(
                name="exp_n_steps",
                suggest_type="pow2_int",
                low=3,
                high=10,
                target="n_steps",
            ),
            SB3OptunaParamConfig(
                name="gamma_eps",
                suggest_type="float",
                low=0.0001,
                high=0.1,
                log=True,
                one_minus=True,
                target="gamma",
            ),
            SB3OptunaParamConfig(
                name="activation",
                suggest_type="categorical",
                choices=["tanh", "relu"],
                value_mapping={"tanh": nn.Tanh, "relu": nn.ReLU},
                target="policy_kwargs.activation_fn",
            ),
        ]
        params = sample_params_from_config(trial, tune_params)
        self.assertEqual(params["n_steps"], 32)
        self.assertEqual(params["gamma"], 0.99)
        self.assertIs(params["policy_kwargs"]["activation_fn"], nn.ReLU)

    def test_split_sampled_params_and_deep_update(self) -> None:
        sampled = {
            "learning_rate": 1e-4,
            "wrapper": {"max_steps": 50},
            "algo_kwargs": {"batch_size": 512},
        }
        algo, wrapper = split_sampled_params(
            sampled, base_wrapper_kwargs={"dense_scale": 1.0}
        )
        self.assertEqual(algo, {"learning_rate": 1e-4, "batch_size": 512})
        self.assertEqual(wrapper, {"max_steps": 50})

        base = {"nested": {"a": 1, "b": 2}, "flat": 3}
        update = {"nested": {"b": 20, "c": 30}}
        res = deep_update(base, update)
        self.assertEqual(res["nested"], {"a": 1, "b": 20, "c": 30})
        self.assertEqual(res["flat"], 3)

    def test_replicate_config_templates_and_seeds(self) -> None:
        class DummyWrapper(gym.Wrapper):
            pass

        class DummyPipelineReader(SB3PipelineConfigReaderClass):
            def _to_wrapper_config(
                self, replicate_signature: str = ""
            ) -> WrapperConfig:
                return WrapperConfig(
                    wrapper_class=DummyWrapper,
                    wrapper_kwargs={
                        "model_path": f"out/{replicate_signature}/model.zip",
                        "rep": "{rep_id}",
                    },
                )

        single_reader = DummyPipelineReader.from_yaml(
            "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
        )
        reader = SB3ReplicatePipelineConfigReader[DummyPipelineReader](
            replicate_config=ReplicateConfig(
                num_replicates=3, replicate_signature="rep_{rep_id}"
            ),
            single_pipeline_config=single_reader,
        )
        replicate_config = reader.to_config()
        self.assertLen(replicate_config.ind_pipeline_configs, 3)

        c0 = replicate_config.ind_pipeline_configs[0]
        c1 = replicate_config.ind_pipeline_configs[1]
        c2 = replicate_config.ind_pipeline_configs[2]
        self.assertIsNotNone(c0.wrapper_config)
        self.assertIsNotNone(c1.wrapper_config)
        assert c0.wrapper_config is not None
        assert c1.wrapper_config is not None
        self.assertEqual(
            c0.wrapper_config.wrapper_kwargs["model_path"],
            "out/rep_0/model.zip",
        )
        self.assertEqual(
            c1.wrapper_config.wrapper_kwargs["model_path"],
            "out/rep_1/model.zip",
        )

        s0 = c0.algo_config.algo_kwargs["seed"]
        s1 = c1.algo_config.algo_kwargs["seed"]
        s2 = c2.algo_config.algo_kwargs["seed"]
        self.assertEqual(s1, s0 + 10)
        self.assertEqual(s2, s0 + 20)

    def test_sample_params_from_config_range_indices(self) -> None:
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        tune_params = [
            SB3OptunaParamConfig(
                name="L_gain_min",
                target="wrapper_kwargs.spec_rep_args.args.L_gain.range.0",
                suggest_type="float",
                low=10.0,
                high=30.0,
            ),
            SB3OptunaParamConfig(
                name="L_gain_max",
                target="wrapper_kwargs.spec_rep_args.args.L_gain.range.1",
                suggest_type="float",
                low=60.0,
                high=120.0,
            ),
        ]
        sampled = sample_params_from_config(trial, tune_params)
        l_gain_range = sampled["wrapper_kwargs"]["spec_rep_args"]["args"][
            "L_gain"
        ]["range"]
        self.assertLen(l_gain_range, 2)
        self.assertLess(l_gain_range[0], l_gain_range[1])

    def test_sample_params_from_config_invalid_range_pruned(self) -> None:
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        tune_params = [
            SB3OptunaParamConfig(
                name="bad_min",
                target="wrapper_kwargs.spec_rep_args.args.L_gain.range.0",
                suggest_type="float",
                low=100.0,
                high=100.0,
            ),
            SB3OptunaParamConfig(
                name="bad_max",
                target="wrapper_kwargs.spec_rep_args.args.L_gain.range.1",
                suggest_type="float",
                low=10.0,
                high=10.0,
            ),
        ]
        with self.assertRaises(optuna.TrialPruned):
            sample_params_from_config(trial, tune_params)

    def test_categorical_complex_choices_sampling_and_decoding(self) -> None:
        db_path = self.temp_dir / "complex.db"
        study = optuna.create_study(
            storage=f"sqlite:///{db_path}", study_name="complex"
        )
        trial = study.ask()
        tune_params = [
            SB3OptunaParamConfig(
                name="activation",
                target="policy_kwargs.activation",
                suggest_type="categorical",
                choices=["relu", "tanh"],
                value_mapping={"relu": "mapped_relu", "tanh": "mapped_tanh"},
            ),
        ]
        sampled = sample_params_from_config(trial, tune_params)
        self.assertIn(
            sampled["policy_kwargs"]["activation"],
            ["mapped_relu", "mapped_tanh"],
        )
        study.tell(trial, 1.0)
        decoded = decode_trial_params(study.best_params, tune_params)
        self.assertIn(decoded["activation"], ["mapped_relu", "mapped_tanh"])

    def test_trial_loss_callback_reports_metric(self) -> None:
        mock_trial = mock.create_autospec(
            optuna.Trial, instance=True, spec_set=True
        )
        mock_trial.should_prune.return_value = False
        cb = TrialLossCallback(
            trial=mock_trial, metric="loss/total", eval_freq=2
        )
        cb.locals = {"stats": {"loss/total": 0.25}}
        cb.n_calls = 2
        res = cb._on_step()

        self.assertTrue(res)
        self.assertEqual(cb.last_loss, 0.25)
        mock_trial.report.assert_called_with(0.25, 1)

    def test_trial_eval_callback_timestep_triggering(self) -> None:
        mock_trial = mock.create_autospec(
            optuna.Trial, instance=True, spec_set=True
        )
        mock_trial.should_prune.return_value = False
        mock_env = mock.create_autospec(VecEnv, instance=True, spec_set=True)

        cb = TrialEvalCallback(
            eval_env=mock_env, trial=mock_trial, eval_freq=1000
        )
        cb._evaluate = mock.create_autospec(
            cb._evaluate, spec_set=True, return_value=True
        )
        cb.last_mean_reward = 42.0

        cb.n_calls = 1
        cb.num_timesteps = 500
        self.assertTrue(cb._on_step())
        self.assertEqual(cb._evaluate.call_count, 0)

        cb.n_calls = 2
        cb.num_timesteps = 1100
        self.assertTrue(cb._on_step())
        self.assertEqual(cb._evaluate.call_count, 1)
        mock_trial.report.assert_called_with(42.0, 1)

    def test_trial_eval_callback_on_training_end_fallback(self) -> None:
        mock_trial = mock.create_autospec(
            optuna.Trial, instance=True, spec_set=True
        )
        mock_env = mock.create_autospec(VecEnv, instance=True, spec_set=True)

        cb = TrialEvalCallback(
            eval_env=mock_env, trial=mock_trial, eval_freq=10_000
        )
        cb.last_mean_reward = -float("inf")

        def fake_eval() -> bool:
            cb.last_mean_reward = 25.0
            return True

        cb._evaluate = mock.create_autospec(
            cb._evaluate, spec_set=True, side_effect=fake_eval
        )
        cb._on_training_end()

        self.assertEqual(cb._evaluate.call_count, 1)
        mock_trial.report.assert_called_with(25.0, 1)

    def test_sb3_pipeline_optimize_smoke(self) -> None:
        config = self._create_pipeline_config()
        db_path = self.temp_dir / "study.db"
        config.optuna_config = SB3OptunaConfig(
            storage_url=f"sqlite:///{db_path}",
            study_name="sb3_smoke_optuna",
            n_trials=1,
            n_jobs=1,
            n_startup_trials=0,
            n_warmup_steps=0,
            n_evaluations=1,
            n_eval_episodes=1,
            total_timesteps=1,
            tune_params=[
                SB3OptunaParamConfig(
                    name="learning_rate",
                    suggest_type="float",
                    low=1e-5,
                    high=1e-2,
                    log=True,
                )
            ],
            dashboard=SB3OptunaDashboardConfig(launch=False),
        )

        pipeline = SB3Pipeline(config=config, verbose=False)
        study = pipeline.optimize()

        self.assertLen(study.trials, 1)
        self.assertEqual(study.study_name, "sb3_smoke_optuna")
        self.assertTrue(db_path.exists())

    def test_sb3_pipeline_optimize_initializes_study_before_dashboard(
        self,
    ) -> None:
        config = self._create_pipeline_config()
        config.optuna_config = SB3OptunaConfig(
            storage_url="sqlite:///study_order_test.db",
            n_trials=1,
            dashboard=SB3OptunaDashboardConfig(launch=True),
            tune_params=[],
            sample_params_fn=lambda trial: {},
        )
        pipeline = SB3Pipeline(config=config, verbose=False)
        call_order: list[str] = []

        class DummyStudy:
            def optimize(self, *args: Any, **kwargs: Any) -> None:
                return None

        fake_study = DummyStudy()

        self.enter_context(
            mock.patch(
                "rl_pipeline.experiment.optuna.create_study",
                side_effect=lambda *a, **kw: (
                    call_order.append("create_study"),
                    fake_study,
                )[1],
            )
        )
        self.enter_context(
            mock.patch(
                "rl_pipeline.experiment.optuna.launch_dashboard",
                side_effect=lambda *a, **kw: (
                    call_order.append("launch_dashboard"),
                    SimpleNamespace(poll=lambda: None),
                )[1],
            )
        )

        study = pipeline.optimize()
        self.assertEqual(call_order, ["create_study", "launch_dashboard"])
        self.assertIs(study, fake_study)

    def test_sb3_pipeline_optimize_uses_process_backend(self) -> None:
        config = self._create_pipeline_config()
        config.optuna_config = SB3OptunaConfig(
            storage_url="sqlite:///study_process_mode.db",
            n_trials=5,
            n_jobs=3,
            parallel_backend="process",
            dashboard=SB3OptunaDashboardConfig(launch=False),
            tune_params=[],
            sample_params_fn=lambda trial: {},
        )
        pipeline = SB3Pipeline(config=config, verbose=False)
        optimize_calls: list[dict[str, int]] = []
        studies: list[object] = []

        class FakeStudy:
            def optimize(
                self, objective: Any, n_trials: int, timeout: Any, n_jobs: int
            ) -> None:
                optimize_calls.append({"n_trials": n_trials, "n_jobs": n_jobs})

        class FakeProcess:
            def __init__(self, target: Any, args: Any) -> None:
                self._target = target
                self._args = args
                self.exitcode = 0
                self.pid = 12345

            def start(self) -> None:
                self._target(*self._args)

            def join(self) -> None:
                return None

        class FakeContext:
            Process = FakeProcess

        self.enter_context(
            mock.patch(
                "rl_pipeline.experiment.optuna.create_study",
                side_effect=lambda *a, **kw: (
                    studies.append(FakeStudy()),
                    studies[-1],
                )[1],
            )
        )
        self.enter_context(
            mock.patch(
                "multiprocessing.get_all_start_methods", return_value=["fork"]
            )
        )
        self.enter_context(
            mock.patch(
                "multiprocessing.get_context", return_value=FakeContext()
            )
        )

        result = pipeline.optimize()
        self.assertIs(result, studies[-1])
        self.assertLen(studies, 5)
        self.assertEqual(
            optimize_calls,
            [
                {"n_trials": 2, "n_jobs": 1},
                {"n_trials": 2, "n_jobs": 1},
                {"n_trials": 1, "n_jobs": 1},
            ],
        )

    def test_sb3_pipeline_optimize_saves_trial_model_artifacts(self) -> None:
        config = self._create_pipeline_config()
        config.optuna_config = SB3OptunaConfig(
            storage_url=f"sqlite:///{self.temp_dir / 'artifact_test.db'}",
            n_trials=1,
            n_jobs=1,
            n_startup_trials=0,
            n_warmup_steps=0,
            n_evaluations=1,
            n_eval_episodes=1,
            total_timesteps=1,
            tune_params=[],
            sample_params_fn=lambda trial: {},
            dashboard=SB3OptunaDashboardConfig(launch=False),
        )
        pipeline = SB3Pipeline(config=config, verbose=False)
        trial_best_dirs: list[str | None] = []

        class FakeTrialEvalCallback:
            def __init__(
                self, *args: Any, best_model_save_path: str, **kwargs: Any
            ) -> None:
                trial_best_dirs.append(best_model_save_path)
                self.is_pruned = False
                self.last_mean_reward = 1.0

        self.enter_context(
            mock.patch(
                "rl_pipeline.sb3.pipeline.TrialEvalCallback",
                FakeTrialEvalCallback,
            )
        )

        study = pipeline.optimize()
        trial = study.trials[0]
        expected_dir = (
            Path(config.save_config.model_save_dir)
            / "optuna_trials"
            / "trial_0"
        )

        self.assertEqual(trial_best_dirs, [str(expected_dir)])
        self.assertEqual(trial.user_attrs["artifact_dir"], str(expected_dir))
        self.assertTrue((expected_dir / "final_model.zip").exists())

    def test_sb3_pipeline_optimize_wrapper_param_tuning(self) -> None:
        class KwargRecorderWrapper(gym.Wrapper):
            recorded_kwargs: ClassVar[dict[str, Any]] = {}

            def __init__(self, env: gym.Env[Any, Any], **kwargs: Any) -> None:
                super().__init__(env)
                KwargRecorderWrapper.recorded_kwargs.update(kwargs)

        config = self._create_pipeline_config()
        config.wrapper_config = WrapperConfig(
            wrapper_class=KwargRecorderWrapper,
            wrapper_kwargs={"max_steps": 10, "base_param": "foo"},
        )
        config.optuna_config = SB3OptunaConfig(
            storage_url=f"sqlite:///{self.temp_dir / 'wrapper.db'}",
            study_name="wrapper_test",
            n_trials=1,
            n_jobs=1,
            n_startup_trials=0,
            n_warmup_steps=0,
            n_evaluations=1,
            n_eval_episodes=1,
            total_timesteps=1,
            tune_params=[
                SB3OptunaParamConfig(
                    name="max_steps",
                    target="wrapper_kwargs.max_steps",
                    suggest_type="categorical",
                    choices=[42],
                )
            ],
            dashboard=SB3OptunaDashboardConfig(launch=False),
        )
        pipeline = SB3Pipeline(config=config, verbose=False)
        study = pipeline.optimize()

        self.assertEqual(study.trials[0].params["max_steps"], 42)
        self.assertEqual(
            KwargRecorderWrapper.recorded_kwargs.get("max_steps"), 42
        )

        summary_path = self.temp_dir / "best.yaml"
        best_summary = pipeline.export_best_params(study, out_path=summary_path)
        self.assertEqual(best_summary["best_params"]["max_steps"], 42)
        self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    absltest.main()
