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

    # Optional: dump only on selected training steps (1-based), e.g. steps 1, 2, and 5:
    # actor_rollout_ref.rollout.skip.dump_steps='[1,2,5]'


Configuration Parameters
------------------------
- **skip.enable**: Enable or disable RolloutSkip.
- **skip.dump_dir**: Root directory to save cached rollout data.
- **skip.max_dump_step**: When ``dump_steps`` is unset or empty, dump/load while ``train_step <= max_dump_step``.
- **skip.dump_steps**: Optional explicit list of 1-based steps to dump/load. If non-empty, only those steps match; otherwise the ``max_dump_step`` window applies. Use null or ``[]`` for default behavior.
- **skip.action**: Applies on **non-dump** steps only. ``cache`` — always generate. ``repeat`` — reuse rollout files in round-robin over ``genstep_*`` dirs that were saved on earlier dump steps. ``repeat_last`` — reuse only the latest such dir. On dump steps, RolloutSkip always tries disk load first, then generate+dump if needed.


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