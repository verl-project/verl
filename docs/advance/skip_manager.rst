SkipManager: Skip everything in the RL pipeline.
===========

Last updated: 2026-05-23

.. contents:: :local:
   :depth: 1

1. Overview
-----------

**SkipManager** (``verl.utils.skip.SkipManager``) is a general-purpose framework for **skipping
selected steps** in verl training flows. By bypassing expensive stages on configured steps, it helps
save **time**, **memory**, or other resources and improves **developer iteration speed** during
debugging and experimentation.

Skip behavior is centralized under the top-level Hydra key ``skip``. Modules register by **role**
(for example ``"rollout"`` or ``"async_rollout"``) and are attached with
``@SkipManager.annotate(role=...)``. Each role declares which integer **steps** in config are
eligible for skip logic. **Today only rollout-related roles are implemented**; the same mechanism
can be extended to other pipeline stages (see section 5).

Typical use cases
~~~~~~~~~~~~~~~~~

SkipManager is intended for development workflows where repeating full training is costly:

1. **Faster iteration**: skip heavy stages on chosen steps (e.g. generation) while exercising the
   rest of the pipeline.
2. **Deterministic replay**: cache and reload intermediate results to reproduce a prior run on
   specific steps.
3. **Resource savings**: avoid recomputing or holding large tensors when bisecting bugs or tuning
   downstream logic.

The built-in ``rollout`` / ``async_rollout`` modules apply this to sequence generation; other
roles can follow the same pattern as they are added.

Supported entry points today
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Training entry
     - Skip role / config
     - Status
   * - ``main_ppo.py`` (``RayPPOTrainer``)
     - ``skip.rollout``
     - **Supported**
   * - ``main_ppo_sync.py`` (TransferQueue + ReplayBuffer)
     - ``skip.rollout``
     - **Not supported** (see section 3)
   * - ``fully_async_main`` (``FullyAsyncRollouter``)
     - ``skip.async_rollout``
     - **Supported**


2. Shared configuration (``skip.rollout`` / ``skip.async_rollout``)
---------------------------------------------------------------------

Both roles use the same Hydra fields (``RolloutSkipConfig`` / ``AsyncRolloutSkipConfig`` in
``verl/utils/skip/config.py``). Defaults live in ``verl/trainer/config/ppo_trainer.yaml`` under
``skip.rollout`` and ``skip.async_rollout``.

Parameters
~~~~~~~~~~

- **enable** (bool): Master switch for this role.
- **dump_dir** (str): Root directory for cached ``DataProto`` shards (``~`` is expanded).
- **steps** (list[int]): Steps on which skip logic is *eligible*. Outside this list, the decorated
  function always runs normally.

  - For ``skip.rollout``: trainer **global_steps** (via ``SkipManager.set_step``).
  - For ``skip.async_rollout``: the feed-order index parsed from ``sample_id`` (see section 4) —
    **not** trainer ``global_steps``.

- **action** (``cache`` \| ``repeat``):

  - **cache**: If a valid dump exists for the current step, load it and skip generation; otherwise
    run generation and write under that step directory.
  - **repeat**: If any valid dump exists, load from a **substitute** step chosen by the algorithm
    below; otherwise run generation and dump as usual.

.. note::

   Only ``cache`` and ``repeat`` are validated in config today, even though ``SkipAction`` in
   ``verl.utils.skip.base_skip`` lists additional enum values for future modules.

``repeat`` step selection (``RolloutSkip._find_latest_step``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``action=repeat`` and the current step directory is missing or incomplete:

1. If the directory for the **current** step is valid, use the current step.
2. Else use the **largest** available step **strictly less than** the current step.
3. Else use the **smallest** available step **strictly greater than** the current step.
4. If no valid dump exists, skip does not apply: the wrapped function runs and may dump afterward.

``repeat`` does **not** guarantee the cached batch matches the current prompt or trainer step—use
it for debugging and iteration, and prefer ``cache`` when you need step-aligned replay.

Hydra CLI examples
~~~~~~~~~~~~~~~~~~

Colocated PPO (``skip.rollout``):

.. code-block:: bash

   skip.rollout.enable=True
   skip.rollout.dump_dir=/path/to/rollout_dump
   skip.rollout.steps=[1,2,3,10]
   skip.rollout.action=cache

Fully async (``skip.async_rollout``):

.. code-block:: bash

   skip.async_rollout.enable=True
   skip.async_rollout.dump_dir=/path/to/rollout_dump
   skip.async_rollout.steps=[1,2,3,4,5]
   skip.async_rollout.action=cache

To pass a long step list from **bash** only (not valid inside static YAML):

.. code-block:: bash

   skip.async_rollout.steps="[$(seq -s, 1 128)]"

On-disk layout
~~~~~~~~~~~~~~

.. code-block:: text

   {dump_dir}/{experiment_name}_{project_name}/
       └── GBS{gbs}_N{n}_in{prompt_len}_out{response_len}/
           ├── {step}/
           │   ├── gen_batch.dp
           │   └── meta.json
           └── ...

- **experiment_name** / **project_name**: from ``trainer.experiment_name`` and
  ``trainer.project_name`` in the run config.
- **gbs**, **n**, **prompt_len**, **response_len**: from ``data.gen_batch_size`` (or train batch
  size), ``actor_rollout_ref.rollout.n``, ``data.max_prompt_length``, and
  ``data.max_response_length``.

Caches from colocated ``main_ppo`` (larger **GBS**) and fully async streaming (typically **GBS=1**)
are generally **not** interchangeable unless these metadata match.

Minimal workflow (``cache``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **First run** with ``enable=True``, ``action=cache``, and ``steps`` listing the steps you care
   about. Empty ``dump_dir`` → generation runs and writes ``gen_batch.dp`` + ``meta.json`` per step.
2. **Second run** with the same config and compatible trainer metadata → listed steps load from
   disk instead of regenerating.
3. **Partial caches** (some step dirs missing): those steps regenerate on the next run; other steps
   still load if present.

Relationship to legacy RolloutSkip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If **both** ``skip.rollout.enable`` and legacy ``actor_rollout_ref.rollout.skip.enable`` are true,
SkipManager emits a ``DeprecationWarning`` and **forces** the legacy flag to ``False`` so only one
mechanism runs.


3. Rollout quick start (``rollout`` role)
-----------------------------------------

Use ``skip.rollout`` when training with ``main_ppo.py`` / ``RayPPOTrainer`` and the standard
``AgentLoopManager.generate_sequences`` path. Configuration fields and ``cache`` / ``repeat``
semantics are in section 2.

**``main_ppo.py`` (supported)**

- ``RayPPOTrainer.fit()`` calls ``SkipManager.init(self.config)`` and
  ``SkipManager.set_step(self.global_steps)`` each training step.
- ``AgentLoopManager.generate_sequences`` is decorated with
  ``@SkipManager.annotate(role="rollout")``.

**``main_ppo_sync.py`` (not supported yet)**

``main_ppo_sync`` replaces the Agent Loop integration with ``AgentLoopManagerTQ``. The main reason
rollout skip is not supported today is **logic coupling** in
``AgentLoopManagerTQ.generate_sequences``: it not only drives sequence generation, but also marks
samples in the ReplayBuffer and **writes generated data into TransferQueue (TQ)**. Skipping
``generate_sequences`` would therefore skip both generation and the TQ handoff, which breaks the
downstream training loop that consumes data from TQ.

Decoupling “generate” from “enqueue to TQ” is non-trivial under the current design, so SkipManager
adaptation for ``main_ppo_sync`` is **deferred** until the TransferQueue-based training path is
further stabilized.


4. Fully async quick start (``async_rollout`` role)
---------------------------------------------------

In :doc:`advance/fully_async`, Trainer and Rollouter run in separate processes. Rollout generation
happens on the Rollouter via streaming single-sample dispatch. Use ``skip.async_rollout`` (not
``skip.rollout``) when launching ``fully_async_main``. Shared Hydra fields and on-disk layout are
in section 2.

.. important::

   In ``async_rollout``, a step is **not** the trainer timeline. It is only the **prompt request /
   feed order** on the Rollouter: the monotonic index in ``sample_{epoch}_{index}`` when
   ``FullyAsyncRollouter`` enqueues the next prompt. Under concurrent rollout, completion order can
   differ from feed order; do not treat these indices as trainer ``global_steps`` or parameter-sync
   boundaries when configuring ``skip.async_rollout.steps``.

Step key from ``sample_id``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each fed sample carries an id of the form ``sample_{epoch}_{index}`` (for example
``sample_0_42``). The integer matched against ``skip.async_rollout.steps`` and used for on-disk
directories is the **last segment** — Rollouter feed-order index at enqueue time.

**Wiring**

- ``FullyAsyncRollouter`` calls ``SkipManager.init(self.config)`` in the Rollouter process.
- ``FullyAsyncAgentLoopManager.generate_sequences_single`` is decorated with
  ``@SkipManager.annotate(role="async_rollout")`` and receives ``sample_id`` for online step
  resolution.


5. Design and implementation
----------------------------

SkipManager API
~~~~~~~~~~~~~~~

``SkipManager`` (``verl.utils.skip.skip_manager``) is a class-level registry:

- **``init(config)``**: Parse ``config.skip`` into ``SkipManagerConfig``, instantiate one skip module
  per registered role, and store them in ``SkipManager.skip_instances``.
- **``set_step(step: int)``**: Set ``SkipManager.step`` for roles with ``support_online_step =
  False`` (trainer ``global_steps`` in ``main_ppo``).
- **``annotate(role, **kwargs)``**: Decorator factory for sync or async functions.

Decorator flow
~~~~~~~~~~~~~~

.. code-block:: text

   call decorated function
        │
        ▼
   skip disabled or role missing? ──yes──► run original function
        │no
        ▼
   resolve step (set_step vs extract_step)
        │
        ▼
   step ∉ config.steps? ──yes──► run original function
        │no
        ▼
   meet_precondition (cache/repeat)? ──yes──► warp_function (load cache)
        │no
        ▼
   run original function → prepare_data (dump)

BaseSkip interface
~~~~~~~~~~~~~~~~~~

Each skip module subclasses ``BaseSkip`` (``verl.utils.skip.base_skip``) and registers via
``@register_skip("role_name")``.

- **``support_actions``**: Allowed ``SkipAction`` values for this module.
- **``support_online_step``**: When ``True``, use ``extract_step`` per call instead of
  ``SkipManager.step``.

Instance methods: ``is_enabled``, ``meet_precondition``, ``warp_function``, ``prepare_data``, and
``extract_step`` (required when ``support_online_step`` is ``True``).

``RolloutSkip`` / ``AsyncRolloutSkip`` (``verl.utils.skip.rollout_skip``) implement generation
caching for the ``rollout`` and ``async_rollout`` roles.

Intercepted functions
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 34 28 22

   * - Role
     - Decorated function
     - Defined in
     - Step source
   * - ``rollout``
     - ``AgentLoopManager.generate_sequences``
     - ``verl/experimental/agent_loop/agent_loop.py``
     - ``SkipManager.set_step`` → trainer ``global_steps``
   * - ``async_rollout``
     - ``FullyAsyncAgentLoopManager.generate_sequences_single``
     - ``verl/experimental/fully_async_policy/fully_async_rollouter.py``
     - ``extract_step`` → ``sample_id`` suffix → **prompt feed order**

**``rollout``** wraps the full batch Agent Loop RPC (chunk dispatch, concat, timing) as one skip
unit.

**``async_rollout``** wraps one streaming sample's ``generate_sequences_single(self, prompts,
sample_id)`` so concurrent samples resolve step independently.

Step resolution: ``set_step`` vs ``support_online_step``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See section 2 for ``steps`` semantics per role.

- **Shared ``SkipManager.step``**: One class-level slot per process. Fits sequential trainer loops
  (``main_ppo``): ``set_step(global_steps)`` before rollout.
- **Online step**: ``AsyncRolloutSkip`` sets ``support_online_step = True`` and parses
  ``sample_id`` on each call so in-flight async samples do not share a single counter. For
  ``repeat``, ``RolloutSkip`` recomputes ``_find_latest_step`` on every ``meet_precondition`` and
  ``warp_function`` call (no shared mutable step field on the skip instance).

Extending with custom skip modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Subclass ``BaseSkip`` from ``verl.utils.skip.base_skip``.
2. Decorate the class with ``@register_skip("your_role_name")``.
3. Add a matching field under ``SkipManagerConfig``.
4. Attach ``@SkipManager.annotate(role="your_role_name")``. For concurrent pipelines, prefer
   ``support_online_step = True`` and pass step identity through call arguments.
