# NeoProto integration

Last updated: 08/19/2026.

NeoProto is the default data representation for the V0 `RayPPOTrainer`. Batch
payloads stay in a storage engine while the driver manipulates schemas,
references, and index views. There is no runtime classic/NeoProto selector;
storage selection remains independent and defaults to the Ray object store.

## Public compatibility API

Existing code can continue to import and construct `verl.DataProto`:

```python
from verl import DataProto

batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
```

This public `DataProto` is a NeoProto-backed compatibility view. It preserves
the established `batch`, `non_tensor_batch`, `meta_info`, selection, concat,
repeat, padding, and construction APIs, but it does not use the former eager
TensorDict transport path.

The old concrete implementation remains available as
`verl.LegacyDataProto` (and historically as `verl.protocol.DataProto`) for
explicit compatibility checks. The trainer never selects it.

## Controller boundary

`single_controller` remains the scheduling and RPC layer. It does not know
about a trainer data-plane strategy. After chunking, it invokes the batch hook
unconditionally:

```python
batch_data.prepare_dispatch(chunked_arg)
```

The NeoProto-backed view attaches rank-local object-store ref tables before Ray
serialization. Generic `chunk` and `concat` behavior stays in `BatchData`.

## Worker boundary

Transient RPC controls are attached to a short-lived ref view through
`prepare_worker_request`; the long-lived trainer batch is not mutated. Worker
results are selected and renamed through `collect_worker_output`, without
materializing them on the driver.

Engine entry points continue to use `run_engine_batch(data, impl, spec)`.
NeoProto materializes only the fields declared by `EngineBatchSpec`, converts
them into the no-padding TensorDict expected by existing compute engines,
restores output padding, and returns a ref-backed result. PPO, reward,
advantage, actor, critic, and rollout algorithms are unchanged.

Unexpected non-Neo worker output and `NEO_BRIDGE_FULL_MATERIALIZE=1` are
fail-closed errors. The removed `trainer.data_plane`, `trainer.use_neoproto`,
and `trainer.neoproto_strict_mode` settings must not be passed by launch
scripts.

## Current scope

- The synchronous V0 PPO trainer is the validated integration target.
- NeoProto storage and worker-boundary correctness must be validated before
  performance benchmarking.
- Direct users of `verl.protocol.DataProto` are on the explicit legacy path and
  should migrate to `from verl import DataProto`.
