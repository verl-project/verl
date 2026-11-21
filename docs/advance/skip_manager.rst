SkipManager: Cache and Replay Rollouts
=====================================

Last updated: 11/10/2026.

``skip.rollout_tq`` caches sampled training trajectories from TransferQueue and replays
them in place of regular training-batch generation. It is integrated into all V1 trainer modes: ``sync``,
``colocate_async``, and ``separate_async``.

Configuration
-------------

.. code-block:: yaml

   skip:
     rollout_tq:
       enable: true
       dump_dir: /path/to/rollout_dump
       steps: [1, 2, 3]
       action: cache

* ``enable``: enable caching and replay. The default is ``false``.
* ``dump_dir``: root directory for cached trajectories.
* ``steps``: trainer ``global_steps`` at which caching or replay is active.
  An empty list disables caching and replay for all steps.
* ``action``: ``cache`` or ``repeat``.

With ``cache``, a complete cache for the exact step is replayed. Otherwise,
normal rollout runs and the sampled training trajectories are saved.

With ``repeat``, the exact step is preferred, followed by the latest complete
cached step before it, then the earliest complete cached step after it. If no
complete cache exists, normal rollout runs and saves the result.

For example, append these overrides to a V1 training command:

.. code-block:: bash

   skip.rollout_tq.enable=True \
   skip.rollout_tq.dump_dir=/shared/rollout_cache \
   'skip.rollout_tq.steps=[1,2,3]' \
   skip.rollout_tq.action=cache

Run once to populate the cache, then rerun with the same cache settings to replay
those steps. Replayed trajectories are not a new policy
rollout and are intended for debugging and profiling.

Submit and sample phases
------------------------

V1 submits prompts separately from sampling completed trajectories. SkipManager
intercepts both phases using ``annotate_tq``:

* **Submit**: ``PPOTrainer._add_batch_to_generate`` advances the dataloader and
  assigns fresh prompt UIDs, including on cache hits. If a cache is available,
  it injects cached trajectories into TransferQueue instead of submitting generation.
* **Sample**: ``ReplayBuffer.sample`` samples normally, then saves the training
  batch if the selected action finds no usable cache. With ``repeat``, a cache
  from another step also prevents saving. Validation
  partitions are not saved.

The trainer sets the current step through ``SkipManager.set_step``. When disabled
or outside the configured steps, the decorators call the original methods unchanged.

Enabling ``skip.rollout_tq`` also disables async warmup submissions, even when
``steps`` is empty. Validation, sampler refill requests, and checkpoint recovery
of in-flight prompts bypass the submit decorator and can still run generation.

Cache layout
------------

.. code-block:: text

   <dump_dir>/
     <experiment_name>_<project_name>/
       GBS<gen_batch_size>_N<n>_in<max_prompt_length>_out<max_response_length>/
         <global_step>/
           0/
             tq_batch.pt
             meta.json
           1/
             tq_batch.pt
             meta.json
           ...

``gen_batch_size`` falls back to ``data.train_batch_size`` when unset.
``n`` is ``actor_rollout_ref.rollout.n``.

The number of inner directories equals
``trainer.v1.<trainer_mode>.parameter_sync_step``, defaulting to one when unset
(as in sync and colocated async). Each sample call that saves a batch writes the
next missing inner directory. A cache is considered complete when every required
inner directory contains both files; this check does not validate their contents.

``tq_batch.pt`` stores trajectory tensors, keys, tags, and the saved step.
``meta.json`` records the step and trajectory count. Keep the model, data schema,
and cache settings compatible when reusing a cache; the directory name does not
encode all training settings.

Replay behavior
---------------

Replay loads and merges the inner batches, groups trajectories by their original
prompt UID, and maps those groups onto the fresh prompt UIDs. If fewer groups or
responses are cached than requested, they are reused cyclically; excess groups
are omitted. Trajectory step tags are updated to the current training step, and
prompt keys are marked ``finished`` so the replay buffer can sample them.

Implementation
--------------

* ``verl/utils/skip/config.py`` defines ``RolloutTqSkipConfig`` and ``SkipManagerConfig``.
* ``verl/utils/skip/skip_manager.py`` manages step state and the submit/sample decorators.
* ``verl/utils/skip/rollout_skip.py`` implements cache storage, selection, and injection.
* ``verl/trainer/ppo/v1/trainer_base.py`` and ``replay_buffer.py`` provide the integration points.

The old V0 and fully async recipe skip implementations remain available in their
pinned verl v0.9.1 dependency.
