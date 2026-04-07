# rl-pipeline
Experiment tools for streamlined RL experiment management.

rl-pipeline provides reusable, configuration-driven components for reinforcement
learning workflows. It is designed to make experiment orchestration,
hyperparameter tuning, and evaluation easier to standardize and reproduce.

## Main Features

- Configuration-first SB3 training and evaluation pipelines.
- Replicate pipeline support for multiple independent runs.
- Optuna-powered hyperparameter optimization with pruning support.
- Persistent study storage for resumable/distributed tuning.
- Optional live monitoring with optuna-dashboard.

## Integrations

- Stable-Baselines3 for RL algorithms and callbacks.
- Optuna for automatic hyperparameter search.
- optuna-dashboard for real-time study monitoring.

## Quick Example

```python
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader

config = SB3PipelineConfigReader.from_yaml("path/to/pipeline.yaml").to_config()
pipeline = SB3Pipeline(config=config)

model = pipeline.train()
eval_result = pipeline.evaluate(checkpoint="best")
```

## Documentation

Detailed usage has been moved to Sphinx-ready docs.

- Overview: [docs/overview.rst](docs/overview.rst)
- Integrations: [docs/integrations.rst](docs/integrations.rst)
- SB3 + Optuna tuning guide: [docs/usage/optuna_tuning.rst](docs/usage/optuna_tuning.rst)
- Citation: [docs/citation.rst](docs/citation.rst)
- Documentation entry point: [docs/index.rst](docs/index.rst)

## Citation

If you use rl-pipeline in research, please cite it as software:

```bibtex
@software{yuasa_rl_pipeline,
  title = {rl-pipeline},
  author = {Yuasa, Mikihisa},
  year = {2026},
  url = {https://github.com/<owner>/rl-pipeline}
}
```
