# Enhanced single controller

Last updated: 09/11/2026.

The enhanced single controller provides a backend-neutral control plane for
creating distributed workers, placing them on resources, invoking methods, and
providing a process-global ObjectStore. Ray remains the default execution backend.
Monarch and TorchStore provide an opt-in path for deployments that run inside a
Monarch job.

This page describes the design, configuration, runtime concepts, and
integration guidance.

## Design

The control plane has five main concepts:

- `Runtime` owns backend resources and worker groups for one process.
- `ResourcePool` describes an ordered, sliceable placement of processes.
- `WorkerGroup` owns a group of workers; `RemoteWorkerGroup` is a non-owning
  invocation view over all or part of that group.
- `RemoteCall` is an in-flight invocation with a deadline fixed at
  submission time.
- `ObjectStore` provides backend-neutral storage to every WorkerGroup.

The execution backend handles resource allocation and transport. Ray uses
placement groups and Ray actors. Monarch uses `HostMesh`, `ProcMesh`, and
`ActorMesh`. Dispatch, collection, rank views, fused roles, topology
compilation, and cleanup ordering stay in the shared control plane.

```text
Runtime
├── Topology ──> ResourcePool
├── ObjectStore
└── WorkerGroup ──> Worker processes
                    └── RemoteCall

RuntimeBackend
├── Ray: PlacementGroup + actors + Ray object store
└── Monarch: HostMesh + ProcMesh + ActorMesh + TorchStore
```

The design follows four ownership rules:

1. verl owns orchestration. It validates topology, compiles placement, creates
   worker groups, dispatches calls, and determines shutdown order.
2. The backend owns deterministic execution. It satisfies compiled placement
   requests, starts processes, transports one invocation to one rank, and
   releases backend resources.
3. The root Runtime owns lifecycle. Attached worker processes may create nested
   worker groups, but they do not start global ObjectStore resources or create
   another root.
4. `ObjectStore` is the only control-plane boundary used by storage-backed data
   planes. Ray and TorchStore references do not enter the shared orchestration
   contract.

### Backend contract

Built-in backends implement a private `RuntimeBackend` protocol. Its operations
provide root resource pools, compile backend placement, select device ranges,
derive imperative pools, create worker groups, construct child attach
configuration, and close backend resources. Each backend also supplies one
submission function that receives resolved per-rank calls and returns a
`RemoteCall`.

The backend contract does not decide which model uses a pool, how ranks are
sliced, or how calls are dispatched and collected. Those decisions remain in
the shared Runtime so Ray and Monarch follow the same control semantics.

### Worker execution contract

Each backend process hosts a `WorkerContainer`. The container constructs the
`Worker`, resolves fused roles, and provides one execution contract:

- asynchronous worker methods run on the actor event loop and may serve
  concurrent requests;
- synchronous methods run in a dedicated single-thread executor to preserve
  SPMD call order;
- dispatch and collection metadata is evaluated in the shared
  `RemoteWorkerGroup`;
- shutdown stops admission and drains pending invocations before releasing the
  worker process.

Ray realizes this contract with generated proxy actors. Monarch uses a
`concurrent_endpoint` on its worker actor. This difference stays below the
shared `WorkerGroup` API.

## Choosing a backend

### Ray

Ray is the default. Existing recipes can continue to derive placement from
`trainer.nnodes` and `trainer.n_gpus_per_node`:

```yaml
runtime:
  backend: ray
  ray: {}
```

```bash
python -m verl.trainer.main_ppo \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8
```

An empty topology is intentional. The Runtime uses declarative placement only
when `topology.models` is non-empty; otherwise it preserves the trainer-derived
Ray placement.

### Monarch with TorchStore

Install the pinned Monarch and TorchStore dependencies through the project
extra:

```bash
pip install -e ".[monarch]"
```

Use the opt-in PPO configuration:

```bash
python -m verl.trainer.main_ppo \
  --config-name=ppo_monarch_neoproto \
  trainer.nnodes=2 \
  trainer.n_gpus_per_node=8
```

The preset selects:

```yaml
runtime:
  backend: monarch
  monarch:
    object_store:
      store_name_prefix: verl_ppo
      timeout_s: 300.0
      strategy: host
```

The V0 trainer uses NeoProto-backed `verl.DataProto` on both backends and has no
`trainer.data_plane` selector. Standalone `DataProto` construction
uses inline references without starting a backend.

The preset also composes `topology/monarch_neoproto.yaml`, which places the
actor, rollout, and critic on one device pool. Monarch starts one global
TorchStore and installs its client in the controller and every WorkerGroup.

`runtime.monarch.job_mode` controls job ownership:

- `current` loads the current job supplied by `monarch apply`. This is the
  recipe default.
- `process` creates and owns a local `ProcessJob`. It is intended for local and
  portable end-to-end runs.

## Declarative topology

A topology maps clusters to device pools and models:

```yaml
topology:
  clusters:
    - name: default
      nnodes: 2
      n_gpus_per_node: 8

  device_pools:
    - name: train
      cluster: default
      nnodes: 2
      n_gpus_per_node: 8

  models:
    - name: actor
      worker: actor
      config_key: actor_rollout_ref
      resource_pool: train

    - name: rollout
      worker: rollout
      config_key: actor_rollout_ref
      resource_pool: train

    - name: critic
      worker: critic
      config_key: critic
      resource_pool: train

```

The Runtime validates the complete topology before allocating backend
resources. Validation includes:

- unique cluster, pool, and model names;
- device-pool capacity within each cluster;
- model references to existing pools;
- valid, non-overlapping model `device_range` values.

Models with the same pool and identical device range are colocated. Disjoint
device ranges split a pool into non-owning views. Partial overlap is rejected.
The topology log printed before worker creation shows the resolved model
placement.

### Extending the standard Monarch topology

The standard preset covers the actor, rollout, and critic used by the default
GAE PPO recipe. A configuration that enables a dedicated reference model,
reward model, teacher model, or standalone rollout group must declare that
model in a replacement topology. Each Runtime process starts its backend's
ObjectStore client automatically.

## Using the Runtime API

`Runtime.from_config` creates the process root. Only one live Runtime may exist
in a process. Use it as a context manager so worker groups, the global
ObjectStore, and backend resources close in the correct order:

```python
from verl.runtime import ClassWithInitArgs, Runtime

runtime_config = {
    "backend": "ray",
    "env_vars": {"MY_WORKER_SETTING": "1"},
    "ray": {},
}

with Runtime.from_config(runtime_config) as runtime:
    pool = runtime.create_resource_pool(
        nnodes=1,
        processes_per_node=8,
        device_type="gpu",
    )
    actor = ClassWithInitArgs(ActorWorker, config)
    actor_wg = runtime.create_worker_group(actor, on=pool)
    output = actor_wg.update_actor(batch)
```

Root-level `runtime.env_vars` is the authoritative environment mapping for
worker processes. Do not duplicate it under `runtime.ray` or
`runtime.monarch`.

When topology declares a model, callers can use its compiled pool:

```python
actor_pool = runtime.model_resource_pool("actor")
actor_wg = runtime.create_worker_group(actor, on=actor_pool)
```

### Single actors and rank views

A single actor is a one-process `WorkerGroup`:

```python
controller_pool = runtime.create_resource_pool(
    nnodes=1,
    processes_per_node=1,
    device_type="cpu",
    on="controller",
)
runner = runtime.create_worker_group(
    ClassWithInitArgs(TaskRunner),
    on=controller_pool,
)
runner.execute_rank_zero_sync("run", config)
```

Rank views do not allocate new resources:

```python
rank_zero = actor_wg.rank(0)
second_tp_group = actor_wg.slice(start=8, size=8)
```

`WorkerGroup` owns the underlying processes. `RemoteWorkerGroup` and rank views
only provide invocation handles; closing a view does not create a second
lifecycle owner.

Runtime tracks groups without keeping them alive. Local rank and fused-role
views retain their parent group; remote projections do not. Garbage collection
of the last local group or view releases the backend workers asynchronously.
Ray actors explicitly created with detached lifetime remain independent of this
collection. Explicit `Runtime.close()` closes live groups and drains pending
backend termination before releasing its dependencies.

### Non-blocking calls and deadlines

Methods retain the dispatch and collection behavior defined by `@register`.
Call `submit` with `blocking=False` to receive a `RemoteCall`:

```python
call = actor_wg.submit(
    "update_actor",
    args=(batch,),
    timeout=300,
    blocking=False,
)
result = call.result()
```

The timeout becomes an absolute deadline when the call is submitted. Delaying
`result()` does not restart the timeout. `RemoteCall.gather(calls)` preserves
input order. `RemoteCall.wait(calls, count=n)` returns another `RemoteCall`
whose result contains the completed and pending calls.

In asynchronous code, await the same object:

```python
result = await call
```

## ObjectStore

Object storage is exposed through the module-level `verl.runtime.put`, `get`,
`delete`, and batch functions. Ray implements them with native ObjectRefs;
Monarch implements them with TorchStore. Backend implementation types are not
part of the user API.

The selected backend installs a process-global client in the controller and
every WorkerGroup. WorkerGroup shutdown does not close that client. The root
Runtime drains its controller client once after all WorkerGroups have stopped,
then closes the backend-owned global store. Worker processes use process-exit
cleanup after their attached Runtime has stopped.

## Child runtimes and nested worker groups

Backend-spawned workers receive an `AttachSpec` containing Runtime identity,
backend configuration, compiled topology, resource pools, and ObjectStore
client configuration. The child creates a process-local `RuntimeContext`,
exposed through:

```python
from verl.runtime import current_runtime

runtime = current_runtime()
```

This allows rollout replicas and other workers to create nested worker groups
without starting a second root Runtime or acquiring ownership of global
ObjectStore resources.

## Compatibility and migration

Existing Ray recipes remain the default. Legacy imports such as
`RayWorkerGroup`, `RayClassWithInitArgs`, and `ResourcePoolManager` remain
available during migration, but new integrations should use the neutral
`Runtime`, `ResourcePool`, and `WorkerGroup` contracts.

The backend registry is private and currently supports built-in backends only.
Integrations should not register third-party Runtime backends against the
internal protocol.

## Testing backend lifecycles

Run the complete controller selection with a fresh Python interpreter per case:

```bash
python -m tests.run_isolated_tests \
  --output /tmp/runtime-tests \
  tests/runtime tests/single_controller
```

The runner retains the collected case manifest, individual pytest logs, JUnit
reports and exit codes. Each test executes its normal fixtures and teardown.
This isolates Monarch's process-global client, which cannot restart after
shutdown in the same interpreter. Live single-host tests use `LocalJob` with
real worker processes. Backend unit fakes cover owned-job cleanup and rollback.
This is a standalone validation entrypoint. The portable 1x8 E2E runner also
creates a private LocalJob context for each Monarch case.

## Current limitations

- Remote work must run as a method on a long-lived `Worker`; there is no
  stateless task API equivalent to decorating a function with `ray.remote`.
- Resource pools are homogeneous rectangular grids. One pool cannot express
  different process counts on different nodes.
- Pool views must select contiguous processes within one node or complete
  contiguous nodes. Ragged and strided selections are not supported.
- `DevicePool.attributes` is preserved but is not enforced for placement.
- Fine-grained NUMA and NIC affinity are outside the current topology model.
- The TRT-LLM asynchronous server and Ray rendezvous remain Ray-specific.
