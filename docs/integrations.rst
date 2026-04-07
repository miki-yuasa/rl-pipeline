Integrations
============

Stable-Baselines3
-----------------

rl-pipeline uses Stable-Baselines3 for algorithm implementations, callback
execution, and policy evaluation.

Optuna
------

rl-pipeline integrates Optuna for hyperparameter optimization with support for:

- Persistent storage backends (for example SQLite, PostgreSQL, MySQL).
- Pruning-enabled evaluation through trial-aware callbacks.
- Config-driven search-space declarations.

optuna-dashboard
----------------

optuna-dashboard can be launched manually or via configuration to monitor study
progress in real time when a persistent storage backend is used.
