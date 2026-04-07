SB3 + Optuna Tuning
===================

The SB3 pipeline supports Optuna optimization through
``SB3Pipeline.optimize()``.

Configure persistent storage
----------------------------

Use a relational storage URL so studies are persisted and can be monitored
live.

.. code-block:: yaml

   optuna_config:
     storage_url: sqlite:///sb3_study.db
     study_name: cartpole_ppo
     direction: maximize
     n_trials: 50
     n_jobs: 1
     n_startup_trials: 5
     n_warmup_steps: 1
     n_evaluations: 2
     n_eval_episodes: 3
     deterministic_eval: true
     total_timesteps: 20000
     dashboard:
       launch: false

     # Optional: declare exactly which params to tune
     tune_params:
       - name: learning_rate
         suggest_type: float
         low: 1e-5
         high: 1e-2
         log: true
       - name: gamma_eps
         suggest_type: float
         low: 1e-4
         high: 1e-1
         log: true
         one_minus: true
         target: gamma
       - name: exponent_n_steps
         suggest_type: pow2_int
         low: 3
         high: 10
         target: n_steps
       - name: activation
         suggest_type: categorical
         choices: [tanh, relu]
         target: policy_kwargs.activation_fn
         value_mapping:
           tanh: torch.nn.Tanh
           relu: torch.nn.ReLU

Run optimization
----------------

.. code-block:: python

   from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader

   config = SB3PipelineConfigReader.from_yaml("path/to/pipeline.yaml").to_config()
   pipeline = SB3Pipeline(config=config)
   study = pipeline.optimize()

   print(study.best_trial.value)
   print(study.best_trial.params)

Monitor with optuna-dashboard
-----------------------------

Manual launch:

.. code-block:: bash

   optuna-dashboard sqlite:///sb3_study.db

Or enable dashboard subprocess launch in config:

.. code-block:: yaml

   optuna_config:
     storage_url: sqlite:///sb3_study.db
     dashboard:
       launch: true
       host: 0.0.0.0
       port: 8080
