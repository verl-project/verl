SkipManager: Rollout Replay and Training Skips
==============================================

Last updated: 11/18/2026.

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

Training skip for inference debugging
------------------------------------

``skip.updates`` and ``skip.train_compute`` are independent roles alongside ``skip.rollout_tq``.
They support all three V1 trainer modes.
Models and workers are still initialized normally. Keep the rollout skip roles disabled to retain
real generation.

.. code-block:: yaml

   skip:
     updates:
       enable: true
       steps: [2, 3]
       action: empty
     train_compute:
       enable: true
       steps: [4]
       action: empty

- ``updates`` skips the complete actor/critic update calls, including training forward/backward
  passes, optimizer steps, and scheduler steps. Training input computation remains.
- ``train_compute`` skips training batch balancing, old/reference log probabilities, critic values,
  advantages/returns, training reward shaping, rollout correction, and the updates inside that
  computation. Reward evaluation and sampler filtering still run; rollout-side log probabilities
  are unaffected. Updates are bypassed even if ``skip.updates.enable=false``, because their inputs
  have not been computed.
- Each role has its own ``enable`` (default ``false``) and ``steps``. ``steps=null`` selects all
  training steps; ``steps=[]`` selects none; ``steps=[2,3]`` selects only steps 2 and 3. Steps are
  positive **trainer global steps**, including after checkpoint resume. In V1, all mini-batches
  within the selected ``parameter_sync_step`` cycle are skipped. Rollout roles still require
  explicit step lists.
- ``action=empty`` is the only supported action: return the input batch without caching or
  fabricating training results. Both roles share ``TrainingSkipConfig`` and the ``TrainingSkip``
  implementation, but ``SkipManager`` creates a separate instance for each role.

In the example, step 1 trains normally, steps 2 and 3 compute training inputs without updating
models, and step 4 bypasses all training computation and updates. The roles preserve rollout,
reward evaluation, queue handling, async warmup, worker sleep/wake, weight synchronization,
validation/checkpoint cadence, and step progression. Trainer steps and synchronization still
advance without optimizer updates.

Append these overrides to an existing training command to skip all training computation on two steps:

.. code-block:: bash

   skip.train_compute.enable=True \
   'skip.train_compute.steps=[2,3]'

Use ``skip.updates.enable=True`` to skip only updates, or set either role's ``steps=null`` to
select all trainer steps.

When either role is enabled, the existing metric logger includes ``training/skip_updates`` and
``training/skip_train_compute``. The former is true when either role skips updates; the latter
tracks only ``train_compute``. All V1 mini-batches within a trainer step share the same selection.
Skipped stages do not fabricate loss or gradient metrics; ``train_compute`` metrics do not require
absent advantages, returns, or values.

Implementation
--------------

* ``verl/utils/skip/config.py`` defines ``RolloutTqSkipConfig`` and ``SkipManagerConfig``.
* ``verl/utils/skip/skip_manager.py`` manages step state and the submit/sample decorators.
* ``verl/utils/skip/rollout_skip.py`` implements cache storage, selection, and injection.
* ``verl/trainer/ppo/v1/trainer_base.py`` and ``replay_buffer.py`` provide the integration points.

The old V0 and fully async recipe skip implementations remain available in their
pinned verl v0.9.1 dependency.
