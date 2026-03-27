RolloutSkip Function Usage Documentation
========================================

Last updated: 2026-03-25

Applicable Scenarios
--------------------
The RolloutSkip utility accelerates RL training by caching and reusing pre-generated rollout data,
avoiding redundant sequence generation during debugging, replay, or fixed-experiment runs.

It is suitable for:

1. Re-running experiments with the same configuration
2. Speeding up training by skipping repeated generation
3. Reproducing rollout results in debugging


API and Usage Example
----------------------

Trainer Adaptation
~~~~~~~~~~~~~~~~~~
RolloutSkip is already supported in ``RayDAPOTrainer`` and ``RayPPOTrainer``.

Example integration:

.. code-block:: python

    from verl.utils.rollout_skip import RolloutSkip

    # Inside trainer.fit()
    rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
    rollout_skip.wrap_generate_sequences()


Basic Configuration
~~~~~~~~~~~~~~~~~~~
Add these parameters to enable RolloutSkip:

.. code-block:: bash

    actor_rollout_ref.rollout.skip.enable=True
    actor_rollout_ref.rollout.skip.dump_dir=/path/to/rollout_dump
    actor_rollout_ref.rollout.skip.max_dump_step=10


Configuration Parameters
------------------------
- **skip.enable**: Enable or disable RolloutSkip.
- **skip.dump_dir**: Root directory to save cached rollout data.
- **skip.max_dump_step**: Maximum number of steps to cache.


Cached Directory Structure
--------------------------
The directory structure is automatically generated to isolate different experiments:

.. code-block:: text

    {dump_dir}/{exp_name}_{project_name}/
    └── GBS{gbs}_N{n}_in{prompt_len}_out{response_len}/
        ├── train_step__gen_step.txt
        ├── genstep_000001/
        │   ├── new_batch.dp
        │   ├── gen_batch.dp
        │   └── meta.json
        └── genstep_000002/


Each ``genstep_*`` folder contains:
- ``new_batch.dp``: Input prompt batch
- ``gen_batch.dp``: Generated response batch
- ``meta.json``: Step metadata