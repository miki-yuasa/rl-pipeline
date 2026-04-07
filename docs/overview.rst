Overview
========

rl-pipeline is a Python library for composing reusable reinforcement learning
experiments with clear configuration boundaries.

Main features
-------------

- Config-driven SB3 pipelines for training, evaluation, and replay recording.
- Experiment manager hooks for external logging systems.
- Built-in Optuna optimization workflow in ``SB3Pipeline.optimize``.
- Configurable hyperparameter search spaces via declarative ``tune_params``.
- Persistent study storage support for distributed and resumable tuning.

Quick start
-----------

.. code-block:: python

   from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader

   config = SB3PipelineConfigReader.from_yaml("path/to/pipeline.yaml").to_config()
   pipeline = SB3Pipeline(config=config)

   model = pipeline.train()
   eval_result = pipeline.evaluate(checkpoint="best")
