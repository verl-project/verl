# NeoProto integration

Last updated: 08/06/2026.

NeoProto is an experimental ref/index data plane for the V0 `RayPPOTrainer`: batch
payloads stay in a storage engine while the controller manipulates schemas, references,
and index views. This page describes how it is wired into the trainer.

The integration rests on a single idea: the data plane is selected once at trainer
startup, so PPO control flow, dispatch, and workers never branch on which batch
implementation is in use.

## Configuration

```yaml
trainer:
  data_plane: classic  # classic | neoproto
```

```bash
python -m verl.trainer.main_ppo trainer.data_plane=neoproto
```

`use_neoproto` is the compatibility field that older launch scripts still pass. It only
upgrades the default `classic` to `neoproto`, and has no effect once `data_plane` names
something else. The full truth table is in the `_resolve_data_plane_name` docstring.

## Selected once

`verl/trainer/ppo/data_plane.py` holds a name-to-strategy registry, and
`_resolve_data_plane_name` is the only place in the repository that reads this
configuration. `RayPPOTrainer` resolves it once during construction:

```python
self.data_plane = build_data_plane(config)
self.data_proto_cls = self.data_plane.data_proto_cls
```

`data_proto_cls` decides which container to create; `data_plane` decides how that
container crosses the worker boundary.

The agent loop worker is a separate Ray actor that must produce batches of the same
type, so it calls `resolve_data_proto_cls(config)` in `__init__` and reads the same
registry. That function resolves the class only and does not call `setup()`, which keeps
the rollout process from taking over the trainer's data-plane initialization.

## The strategy owns the RPC boundary

`PPODataPlane` defines `prepare_inference`, `collect_inference`, `prepare_training`,
`collect_metrics`, `prefetch`, and the materialization counters. The trainer only calls
`self.data_plane.*`:

```python
prepared = self.data_plane.prepare_inference(batch, {"compute_loss": False})
output = self.critic_wg.infer_batch(prepared.payload).get()
return self.data_plane.collect_inference(output, prepared, {"values": "values"}, ...)
```

`ClassicPPODataPlane` converts the DataProto into a de-padded TensorDict on the driver,
then restores padding and rewraps the result on the way back.

`NeoPPODataPlane` sends a ref view instead, and collects results by selecting and
renaming on the ref table without materializing on the driver. It builds a separate
request view per RPC so that transient control fields such as `no_lora_adapter` do not
persist on the long-lived batch or leak into later calls.

## One expression for both containers

Classic `DataProto` provides a set of default hooks that NeoProto overrides only where
behavior genuinely differs:


| hook                       | classic                                     | NeoProto                                 |
| -------------------------- | ------------------------------------------- | ---------------------------------------- |
| `new_like(...)`            | build output with the current concrete type | same                                     |
| `prefetch(...)`            | no-op                                       | populate the lazy cache                  |
| `clear_cache()`            | no-op                                       | drop materialization caches              |
| `prepare_dispatch(chunks)` | no-op                                       | attach rank-local ref tables             |
| `set_control_fields(...)`  | write to `meta_info`                        | write ref-side control fields            |
| `cpu()`                    | `to("cpu")`                                 | return self; refs already resolve to CPU |


Shared code therefore reads identically for both, with no `type(data)(...)` and no type
checks:

```python
output = input_batch.new_like(batch=..., non_tensor_batch=..., meta_info=...)
```



## Dispatch

After chunking, the single controller calls the same hook unconditionally:

```python
batch_data.prepare_dispatch(chunked_arg)
```

Classic DataProto needs no preparation. NeoProto attaches the matching `OBJ_REF` and
`LOCAL_REF` for each rank before Ray serializes the chunks. The ref-table construction
lives in `verl/experimental/neoproto/dispatch.py`.

## Worker adapter

Worker entry points go through `run_engine_batch(data, impl, spec)`. A TensorDict input
is passed straight to the engine. A NeoProto input is delegated to the worker bridge,
which materializes only the required and present optional fields declared by the
`EngineBatchSpec`, converts them into the no-padding TensorDict the engine already
expects, calls the unmodified engine, restores padding, and rewraps the output as a
ref-backed DataProto.

The field lists live next to the engine entry points in
`verl/workers/engine_workers.py` rather than inside the bridge, which keeps the
producer-consumer relationship visible. All of these names are existing public verl
TensorDict fields or worker control fields.

## Why the global DataProto is not replaced

Swapping `verl.DataProto` process-wide is unsafe: modules that already ran
`from verl import DataProto` keep the old reference, so the swap only affects later
imports and leaves two inconsistent DataProto classes alive in the same process. The
integration therefore selects at a single decision point instead of rewriting a global
alias.

## Limitations

- Classic is the default data plane.
- Only the synchronous V0 PPO trainer is covered.
- NeoProto currently uses the Ray object store backend only.
- PPO, reward, advantage, and model engine algorithms are unchanged.

