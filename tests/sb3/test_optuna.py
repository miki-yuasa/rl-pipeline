from pathlib import Path
from types import SimpleNamespace

import optuna
from rl_pipeline.experiment.optuna import build_dashboard_command
from rl_pipeline.sb3 import (
    SB3OptunaConfig,
    SB3OptunaDashboardConfig,
    SB3Pipeline,
    SB3PipelineConfigReader,
)
from rl_pipeline.sb3.config import SB3OptunaParamConfig
from rl_pipeline.sb3.experiment.optuna import (
    filter_algorithm_kwargs,
    sample_params_from_config,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn


def test_filter_algorithm_kwargs_drops_unknown_values():
    from stable_baselines3 import PPO

    kwargs = {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "unknown_param": 123,
    }

    filtered = filter_algorithm_kwargs(PPO, kwargs)

    assert "learning_rate" in filtered
    assert "gamma" in filtered
    assert "unknown_param" not in filtered


def test_dashboard_command_builder():
    command = build_dashboard_command(
        storage_url="sqlite:///study.db",
        host="0.0.0.0",
        port=8080,
    )

    assert command.command == [
        "optuna-dashboard",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "sqlite:///study.db",
    ]


def test_sample_params_from_config_supports_nested_and_mapping():
    trial = optuna.trial.FixedTrial(
        {
            "exp_n_steps": 5,
            "gamma_eps": 0.01,
            "activation": "relu",
        }
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

    assert params["n_steps"] == 32
    assert params["gamma"] == 0.99
    assert params["policy_kwargs"]["activation_fn"] is nn.ReLU


def test_sb3_pipeline_optimize_smoke(tmp_path: Path):
    config = SB3PipelineConfigReader.from_yaml(
        "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
    ).to_config()

    assert config.vec_config is not None
    config.vec_config.vec_env_cls = DummyVecEnv
    config.vec_config.n_envs = 1

    db_path = tmp_path / "study.db"
    config.optuna_config = SB3OptunaConfig(
        storage_url=f"sqlite:///{db_path}",
        study_name="sb3_smoke_optuna",
        n_trials=1,
        n_jobs=1,
        n_startup_trials=1,
        n_warmup_steps=0,
        n_evaluations=1,
        n_eval_episodes=1,
        total_timesteps=64,
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

    assert len(study.trials) == 1
    assert study.study_name == "sb3_smoke_optuna"
    assert db_path.exists()


def test_sb3_pipeline_optimize_initializes_study_before_dashboard(monkeypatch):
    config = SB3PipelineConfigReader.from_yaml(
        "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
    ).to_config()

    assert config.vec_config is not None
    config.vec_config.vec_env_cls = DummyVecEnv
    config.vec_config.n_envs = 1
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
        def optimize(self, *args, **kwargs):
            return None

    fake_study = DummyStudy()

    def fake_create_study(*args, **kwargs):
        call_order.append("create_study")
        return fake_study

    def fake_launch_dashboard(*args, **kwargs):
        call_order.append("launch_dashboard")
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(
        "rl_pipeline.experiment.optuna.create_study",
        fake_create_study,
    )
    monkeypatch.setattr(
        "rl_pipeline.experiment.optuna.launch_dashboard", fake_launch_dashboard
    )

    study = pipeline.optimize()

    assert call_order == ["create_study", "launch_dashboard"]
    assert study is fake_study


def test_sb3_pipeline_optimize_uses_process_backend(monkeypatch):
    config = SB3PipelineConfigReader.from_yaml(
        "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
    ).to_config()

    assert config.vec_config is not None
    config.vec_config.vec_env_cls = DummyVecEnv
    config.vec_config.n_envs = 1
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
        def optimize(self, objective, n_trials, timeout, n_jobs):
            optimize_calls.append({"n_trials": n_trials, "n_jobs": n_jobs})

    def fake_create_study(*args, **kwargs):
        study = FakeStudy()
        studies.append(study)
        return study

    class FakeProcess:
        def __init__(self, target, args):
            self._target = target
            self._args = args
            self.exitcode = 0
            self.pid = 12345

        def start(self):
            self._target(*self._args)

        def join(self):
            return None

    class FakeContext:
        Process = FakeProcess

    monkeypatch.setattr("rl_pipeline.experiment.optuna.create_study", fake_create_study)
    monkeypatch.setattr("multiprocessing.get_all_start_methods", lambda: ["fork"])
    monkeypatch.setattr("multiprocessing.get_context", lambda method: FakeContext())

    result = pipeline.optimize()

    assert result is studies[-1]
    assert len(studies) == 5
    assert optimize_calls == [
        {"n_trials": 2, "n_jobs": 1},
        {"n_trials": 2, "n_jobs": 1},
        {"n_trials": 1, "n_jobs": 1},
    ]


def test_sb3_pipeline_optimize_saves_trial_model_artifacts(monkeypatch, tmp_path: Path):
    config = SB3PipelineConfigReader.from_yaml(
        "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
    ).to_config()

    assert config.vec_config is not None
    config.vec_config.vec_env_cls = DummyVecEnv
    config.vec_config.n_envs = 1

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    config.save_config.model_save_path = str(model_dir / "final_model.zip")

    class DummyAlgorithm:
        def __init__(self, env=None, tensorboard_log=None, device=None, **kwargs):
            self.env = env

        def learn(self, *args, **kwargs):
            return None

        def save(self, path: str):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(f"{path}.zip").write_text("dummy")

    config.algo_config.algorithm = DummyAlgorithm  # type: ignore[assignment]
    config.optuna_config = SB3OptunaConfig(
        storage_url=f"sqlite:///{tmp_path / 'artifact_test.db'}",
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
            self,
            eval_env,
            trial,
            n_eval_episodes,
            eval_freq,
            best_model_save_path,
            deterministic,
            verbose,
        ):
            trial_best_dirs.append(best_model_save_path)
            self.is_pruned = False
            self.last_mean_reward = 1.0

    monkeypatch.setattr(
        "rl_pipeline.sb3.pipeline.TrialEvalCallback", FakeTrialEvalCallback
    )

    study = pipeline.optimize()
    trial = study.trials[0]

    expected_dir = Path(config.save_config.model_save_dir) / "optuna_trials" / "trial_0"
    assert trial_best_dirs == [str(expected_dir)]
    assert trial.user_attrs["artifact_dir"] == str(expected_dir)
    assert trial.user_attrs["best_model_path"] == str(expected_dir / "best_model.zip")
    assert trial.user_attrs["final_model_path"] == str(expected_dir / "final_model.zip")
    assert (expected_dir / "final_model.zip").exists()


def test_split_sampled_params_and_deep_update():
    from rl_pipeline.sb3.experiment.optuna import deep_update, split_sampled_params

    # Target routing test
    sampled = {
        "learning_rate": 1e-4,
        "wrapper": {"max_steps": 50},
        "algo_kwargs": {"batch_size": 512},
    }
    algo, wrapper = split_sampled_params(
        sampled, base_wrapper_kwargs={"dense_scale": 1.0}
    )
    assert algo == {"learning_rate": 1e-4, "batch_size": 512}
    assert wrapper == {"max_steps": 50}

    # Deep update test
    base = {"nested": {"a": 1, "b": 2}, "flat": 3}
    update = {"nested": {"b": 20, "c": 30}}
    res = deep_update(base, update)
    assert res["nested"] == {"a": 1, "b": 20, "c": 30}
    assert res["flat"] == 3


def test_replicate_config_templates_and_seeds(tmp_path: Path):
    import gymnasium as gym
    from rl_pipeline.core import ReplicateConfig
    from rl_pipeline.gymnasium.config import WrapperConfig
    from rl_pipeline.sb3.config_reader import (
        SB3PipelineConfigReader,
        SB3ReplicatePipelineConfigReader,
    )

    class DummyWrapper(gym.Wrapper):
        pass

    class DummyPipelineReader(SB3PipelineConfigReader):
        def _to_wrapper_config(self, replicate_signature: str = ""):
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
    assert len(replicate_config.ind_pipeline_configs) == 3

    # Check template formatting
    assert (
        replicate_config.ind_pipeline_configs[0].wrapper_config.wrapper_kwargs[
            "model_path"
        ]
        == "out/rep_0/model.zip"
    )
    assert (
        replicate_config.ind_pipeline_configs[1].wrapper_config.wrapper_kwargs[
            "model_path"
        ]
        == "out/rep_1/model.zip"
    )

    # Check distinct seeds
    assert (
        replicate_config.ind_pipeline_configs[0].algo_config.algo_kwargs["seed"]
        is not None
    )
    assert (
        replicate_config.ind_pipeline_configs[1].algo_config.algo_kwargs["seed"]
        == replicate_config.ind_pipeline_configs[0].algo_config.algo_kwargs["seed"] + 10
    )
    assert (
        replicate_config.ind_pipeline_configs[2].algo_config.algo_kwargs["seed"]
        == replicate_config.ind_pipeline_configs[0].algo_config.algo_kwargs["seed"] + 20
    )


def test_sb3_pipeline_optimize_wrapper_param_tuning(tmp_path: Path):
    import gymnasium as gym
    from rl_pipeline.gymnasium.config import WrapperConfig

    class KwargRecorderWrapper(gym.Wrapper):
        recorded_kwargs = {}

        def __init__(self, env, **kwargs):
            super().__init__(env)
            KwargRecorderWrapper.recorded_kwargs.update(kwargs)

    config = SB3PipelineConfigReader.from_yaml(
        "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
    ).to_config()

    config.vec_config.vec_env_cls = DummyVecEnv
    config.vec_config.n_envs = 1
    config.wrapper_config = WrapperConfig(
        wrapper_class=KwargRecorderWrapper,
        wrapper_kwargs={"max_steps": 10, "base_param": "foo"},
    )

    db_path = tmp_path / "wrapper_study.db"
    config.optuna_config = SB3OptunaConfig(
        storage_url=f"sqlite:///{db_path}",
        study_name="wrapper_optuna_test",
        n_trials=1,
        n_jobs=1,
        n_startup_trials=1,
        n_warmup_steps=0,
        n_evaluations=1,
        n_eval_episodes=1,
        total_timesteps=32,
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

    assert study.trials[0].params["max_steps"] == 42
    assert KwargRecorderWrapper.recorded_kwargs.get("max_steps") == 42
    assert KwargRecorderWrapper.recorded_kwargs.get("base_param") == "foo"

    # Test export_best_params
    summary_path = tmp_path / "best.yaml"
    best_summary = pipeline.export_best_params(study, out_path=summary_path)
    assert best_summary["best_params"]["max_steps"] == 42
    assert summary_path.exists()


def test_trial_loss_callback_and_minimization(tmp_path: Path):
    from unittest.mock import MagicMock

    import optuna
    from rl_pipeline.sb3.callback import TrialLossCallback

    mock_trial = MagicMock()
    mock_trial.should_prune.return_value = False

    cb = TrialLossCallback(trial=mock_trial, metric="loss/total", eval_freq=2)
    cb.locals = {"stats": {"loss/total": 0.25}}
    cb.n_calls = 2
    res = cb._on_step()

    assert res is True
    assert cb.last_loss == 0.25
    mock_trial.report.assert_called_with(0.25, 1)

    # Test pipeline optimization with loss minimization
    config = SB3PipelineConfigReader.from_yaml(
        "tests/sb3/assets/configs/cartpole_pipeline_config.yaml"
    ).to_config()
    config.learn_config.total_timesteps = 32
    config.vec_config.n_envs = 1

    db_path = tmp_path / "loss_study.db"
    config.optuna_config = SB3OptunaConfig(
        storage_url=f"sqlite:///{db_path}",
        study_name="loss_optuna_test",
        direction="minimize",
        metric="train/loss",
        n_trials=1,
        n_jobs=1,
        n_startup_trials=1,
        n_warmup_steps=0,
        n_evaluations=1,
        total_timesteps=32,
        tune_params=[
            SB3OptunaParamConfig(
                name="learning_rate",
                target="algo_kwargs.learning_rate",
                suggest_type="float",
                low=1e-4,
                high=1e-3,
            )
        ],
        dashboard=SB3OptunaDashboardConfig(launch=False),
    )

    pipeline = SB3Pipeline(config=config, verbose=False)
    study = pipeline.optimize()
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert len(study.trials) == 1
    assert "learning_rate" in study.best_params


def test_categorical_complex_choices_sampling_and_decoding(tmp_path: Path):
    import warnings

    from rl_pipeline.sb3.experiment.optuna import decode_trial_params

    db_path = tmp_path / "complex_choices.db"
    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        study_name="complex_choices_test",
    )
    trial = study.ask()

    tune_params = [
        SB3OptunaParamConfig(
            name="switch_threshold_range",
            target="wrapper_kwargs.spec_rep_args.args.switch_threshold.range",
            suggest_type="categorical",
            choices=[[-30, 0], [-20, 10], [-5, 5]],
        ),
        SB3OptunaParamConfig(
            name="activation",
            target="policy_kwargs.activation",
            suggest_type="categorical",
            choices=["relu", "tanh"],
            value_mapping={"relu": "mapped_relu", "tanh": "mapped_tanh"},
        ),
    ]

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        sampled = sample_params_from_config(trial, tune_params)

    optuna_warnings = [
        w
        for w in recorded_warnings
        if issubclass(w.category, UserWarning)
        and "Choices for a categorical distribution" in str(w.message)
    ]
    assert len(optuna_warnings) == 0

    sampled_range = sampled["wrapper_kwargs"]["spec_rep_args"]["args"][
        "switch_threshold"
    ]["range"]
    assert sampled_range in [[-30, 0], [-20, 10], [-5, 5]]
    assert sampled["policy_kwargs"]["activation"] in ["mapped_relu", "mapped_tanh"]

    study.tell(trial, 1.0)
    decoded = decode_trial_params(study.best_params, tune_params)
    assert decoded["switch_threshold_range"] in [[-30, 0], [-20, 10], [-5, 5]]
    assert decoded["activation"] in ["mapped_relu", "mapped_tanh"]
