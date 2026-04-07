import subprocess
from dataclasses import dataclass

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


@dataclass(frozen=True)
class DashboardCommand:
    command: list[str]

    def as_shell_command(self) -> str:
        return " ".join(self.command)


def create_study(
    storage_url: str,
    study_name: str | None,
    direction: str = "maximize",
    n_startup_trials: int = 5,
    n_warmup_steps: int = 0,
) -> optuna.Study:
    sampler = TPESampler(n_startup_trials=n_startup_trials)
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def build_dashboard_command(
    storage_url: str,
    host: str | None = None,
    port: int | None = None,
) -> DashboardCommand:
    command = ["optuna-dashboard"]
    if host:
        command.extend(["--host", host])
    if port is not None:
        command.extend(["--port", str(port)])
    command.append(storage_url)
    return DashboardCommand(command=command)


def launch_dashboard(
    storage_url: str,
    host: str | None = None,
    port: int | None = None,
) -> subprocess.Popen[str]:
    dashboard_command = build_dashboard_command(
        storage_url=storage_url,
        host=host,
        port=port,
    )
    return subprocess.Popen(dashboard_command.command, text=True)
