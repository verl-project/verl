RolloutSkip Function Usage Documentation
========================================

Last updated: 2026-03-18.

Applicable Scenarios
--------------------

The RolloutSkip functionality is designed to accelerate the rollout process in reinforcement learning training by caching and reusing previously generated sequences. This feature is particularly useful when:

1. You need to repeatedly run experiments with the same configuration

2. You want to save time by avoiding redundant sequence generation to come close to the optimal policy


API and Usage Example
----------------------

2.1 Trainer Adaptation
~~~~~~~~~~~~~~~~~~~~~~

Both`RayDAPOTrainer()` (in `verl/recipe/dapo/dapo_ray_trainer.py`) and `RayPPOTrainer()`(in `verl/trainer/ppo/ray_trainer.py``) have already been adapted.

This is an example of how to patch rollout_skip in RayPPOTrainer.

.. code-block:: python

    #* Import the RolloutSkip class
    from verl.utils.rollout_skip import RolloutSkip

    ...
    class RayPPOTrainer:
        ...
        def fit(self):
            ...

            #* Add code as follow:
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

            ...

            for epoch in range(self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    ...

2.2 Basic Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Then, you should add the following parameters to your config to enable the RolloutSkip feature:

.. code-block:: bash

    actor_rollout_ref.rollout.skip.enable=True \
    actor_rollout_ref.rollout.skip.dump_dir=/path/to/skip_rollout/rollout_dump \
    actor_rollout_ref.rollout.skip.max_dump_step=10 \


Notes
-----

These follow the behavior in ``verl/utils/rollout_skip.py``:
1. **``skip.enable``** — If ``False``, ``RolloutSkip`` returns early from ``__init__`` and does not patch ``generate_sequences``; the trainer must still gate construction on ``enable`` (as in RayPPOTrainer).

2. **``skip.dump_dir``** — Root for dumps (default ``~/.verl/rollout_dump``, expanded at runtime). Must be writable. **Prefer an absolute path** in Ray or multi-process setups: relative paths are resolved against each process's cwd and can point to different locations. Avoid ``/tmp/ray/session*`` (ephemeral); the code warns if the final path lies there.

3. **Subdirectory layout** — Actual cache root is:

       {dump_dir}/{trainer.experiment_name}_{trainer.project_name}/GBS{data.gen_batch_size}_N{rollout.n}_in{max_prompt_length}_out{max_response_length}/

   Each rollout index is a folder ``genstep_000001/``, ``genstep_000002``, … containing ``new_batch.dp``, ``gen_batch.dp``, and ``meta.json``. Changing experiment name, project name, ``data.gen_batch_size`` (GBS), ``rollout.n``, or ``data.max_prompt_length`` / ``max_response_length`` selects a **new** subdirectory, so old caches are not reused.

4. **``skip.max_dump_step``** — Only the first *N* training steps run try-load-then-dump per step; after step *N*, ``action`` decides how rollouts are reused.