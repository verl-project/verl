# Remote Backend: pluggable out-of-process RL backends

Last updated: 07/21/2026.

## Motivation

Some downstream users want to plug their own training + sampling stack
(e.g. an in-house DeepSpeed/vLLM combo) into verl's PPO orchestration
without touching verl's actor / critic / rollout workers or its
checkpoint-engine hot path. Concretely, they want:

- Verl-core to drive the RL control flow (rollout, advantage,
  ``update_actor``, weight sync, metrics) exactly as it does today.
- All GPU compute — model forward, actor optimizer step, log-prob,
  rollout — to happen inside a single out-of-process backend that owns
  its own Ray actors, its own DeepSpeed/FSDP engine, and its own vLLM
  servers.
- Weight sync between train and rollout to happen out-of-band (typically
  CUDA IPC between colocated processes), so verl's per-tensor
  ``send_weights`` / ``receive_weights`` machinery does not run.

The ``RemoteBackend`` abstraction is the seam that makes that possible.
It ships in verl-core under ``verl/remote_backend/`` and integrates with
the V1 trainer via a small subclass ``PPOTrainerRemoteBackend`` selected
by a Hydra group choice ``remote_backend=<name>``. Verl-core itself
carries no concrete backend; a downstream package (Arctic-Platform is
the reference implementation) provides the backend + its forwarder
worker + its rollout replica via ``VERL_USE_EXTERNAL_MODULES``.

The V0 version of this seam lands in
[PR #6422](https://github.com/verl-project/verl/pull/6422); this doc
describes the V1 port that lands in
[PR #7102](https://github.com/verl-project/verl/pull/7102).

## Design

### The ABC

``verl.remote_backend.base.RemoteBackend`` (see
``verl/remote_backend/base.py``) exposes only the lifecycle contract
verl needs to drive:

- ``from_config(main_config, *, handle=None) -> RemoteBackend`` — sole
  public constructor. Called once on the driver with ``handle=None``
  and re-called inside every forwarder worker / rollout replica with
  the driver-side handle so both re-attach instead of building a second
  copy.
- ``reconnect_handle() -> dict[str, Any]`` — serializable handle that
  ``PPOTrainerRemoteBackend`` puts into ``wg_kwargs`` and
  ``LLMServerManager.replica_init_kwargs``.
- ``destroy()`` — idempotent teardown.
- ``update_weights()`` / ``save_checkpoint()`` — async hooks the
  trainer / checkpoint engine calls.
- ``requires_single_forwarder() -> bool`` — whether the trainer should
  assert ``n_gpus_per_node * nnodes == 1``. Backends whose payload
  dispatches through ``Dispatch.ONE_TO_ALL`` need this or their
  ``update_weights`` / ``save_checkpoint`` duplicate against the same
  underlying backend from every forwarder.

Compute/update op signatures (``compute_log_prob``, ``update_actor``,
``generate``) intentionally do **not** live on the ABC. They live on the
concrete backend and its matching per-backend forwarder worker so
different backends can shape their payloads however they need without
growing the base class.

### The registry

``RemoteBackendRegistry`` is a process-wide name-keyed registry. A plugin
registers three things under one name:

1. The ``RemoteBackend`` subclass, via
   ``@RemoteBackendRegistry.register("<name>")`` at class-definition time
   (side effect of importing the adapter module).
2. The ActorRollout forwarder worker class, via
   ``RemoteBackendRegistry.register_worker(name, loader)``. ``loader`` is
   a zero-arg callable that returns the concrete worker class — kept
   lazy so plugin bootstrap does not force imports of vLLM / DeepSpeed /
   tensordict etc.
3. The rollout replica class, via
   ``verl.workers.rollout.replica.RolloutReplicaRegistry.register``.

### The V1 trainer

``PPOTrainerRemoteBackend`` subclasses ``PPOTrainerSync`` and overrides
three extension hooks that ``PPOTrainer`` gained in this PR:

- ``_actor_rollout_wg_extra_kwargs`` — smuggles
  ``{main_config, backend_handle}`` into every actor-rollout worker via
  ``RayClassWithInitArgs``.
- ``_llm_server_replica_init_kwargs`` — forwards the same handle into
  every ``RolloutReplica`` via ``LLMServerManager.replica_init_kwargs``.
- ``_checkpoint_engine_backend`` — returns ``"remote_backend"``.

It also overrides ``_init_resource_pool_mgr`` to (a) install the
plugin's forwarder worker class and (b) flip the pool to CPU-only. The
forwarder does no GPU work; the plugin already owns the GPUs internally,
so double-booking through Ray would starve it ("Total available GPUs 0"
at placement time). ``ResourcePoolManager`` gains a ``use_gpu`` flag
that defaults to ``True`` — no behavior change for non-plugin paths.

Async training is out of scope for this PR; only the sync-colocate
trainer mode is supported today. Async support will land in a follow-up.

### The checkpoint engine

``verl/checkpoint_engine/remote_backend.py`` is a no-op stub registered
under ``"remote_backend"``. It exists only so
``CheckpointEngineManager`` can be constructed against a plugin-driven
backend; ``send_weights`` / ``receive_weights`` are never called on this
path.

``CheckpointEngineManager.update_weights`` short-circuits for
``backend='remote_backend'`` and delegates to
``actor_wg.update_weights(...)``, which the forwarder relays to the
plugin (typically triggering a CUDA-IPC handshake to the rollout
process). After the actor-side ``update_weights`` returns, the manager
also fans ``set_global_steps(global_steps)`` out to any rollout
replicas that implement it, so downstream metrics (staleness, policy
version) can tag each rollout with the correct version.

## Usage

Users select the backend via a Hydra group choice::

    remote_backend=<name>

``_resolve_remote_backend_from_hydra_choice`` in
``verl/trainer/main_ppo.py`` mirrors that onto
``trainer.remote_backend=<name>`` and (unless the user picked a custom
mode) ``trainer.v1.trainer_mode=remote_backend``, which selects
``PPOTrainerRemoteBackend`` through the V1 trainer registry.

The plugin's config file (e.g. ``remote_backend/arctic.yaml``) is
loaded through ``hydra.searchpath``; see the plugin's
``VERL_USE_EXTERNAL_MODULES`` bootstrap module for the full wiring.

## Reference implementation

Arctic-Platform is the reference downstream implementation. The V1
plugin shims (worker, replica, server) plus a from-scratch setup guide
live in
[Snowflake-AI-Research/Arctic-Platform#41](https://github.com/Snowflake-AI-Research/Arctic-Platform/pull/41).

Validation on Qwen3-8B BIRD text-to-SQL, 5 recipe-aligned steps on
8×H200:

| Metric                    | Stock V1 (vLLM+FSDP2) | Arctic V1 + ZoRRo | Speedup   |
| :------------------------ | --------------------: | ----------------: | --------: |
| ``timing_s/update_actor`` |                 380.9 |             166.0 | **2.30×** |
| ``timing_s/step`` (total) |                 731.4 |             438.3 | **1.67×** |

Convergence is preserved: reward means match step-over-step between the
stock and plugin paths.

## Tests

CPU-only unit tests live under ``tests/remote_backend/``:

- Registry decorator behaviour (dedupe, name collision).
- Lazy worker-loader caching.
- Checkpoint-engine registry dispatch.
- Trainer registry dispatch (``get_trainer_cls("remote_backend")``).
- ``_resolve_remote_backend_from_hydra_choice`` behaviour, including
  respecting an explicit user-set ``trainer.v1.trainer_mode``.

Run with::

    pytest tests/remote_backend/

## Files

- ``verl/remote_backend/{base,worker_utils,__init__}.py`` — ABC +
  registry + backend-agnostic tensor/metric helpers.
- ``verl/checkpoint_engine/remote_backend.py`` — no-op checkpoint
  engine stub.
- ``verl/checkpoint_engine/base.py`` — ``update_weights``
  short-circuit + ``set_global_steps`` fan-out.
- ``verl/trainer/ppo/v1/trainer_base.py`` — three extension hooks.
- ``verl/trainer/ppo/v1/trainer_remote_backend.py`` —
  ``PPOTrainerRemoteBackend``.
- ``verl/trainer/main_ppo.py`` —
  ``_resolve_remote_backend_from_hydra_choice``.
- ``verl/workers/rollout/{replica,llm_server}.py`` —
  ``replica_init_kwargs`` plumbing.
- ``verl/single_controller/ray/base.py`` —
  ``ResourcePoolManager.use_gpu``.
