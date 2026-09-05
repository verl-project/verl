# Async collective handle stream contract

Last updated: 09/05/2026.

`verl.utils.collective.AsyncCollectiveHandle` separates communication waiting
from result finalization. It owns references needed by the asynchronous work
and runs the finalizer at most once.

For CUDA collectives, pass the result tensor's device as `consumer_device`.
The first call to `wait_collective()`, `finalize_result()`, or `wait()` binds
the handle to the current CUDA stream on that device. All later calls must use
that same stream. A call from another stream raises an error instead of
silently returning a result whose dependency was established elsewhere.

`Work.wait()` establishes ordering on the bound CUDA stream. It does not mean
that the CPU waited for physical kernel completion. `complete_event`, when
provided, is recorded on the bound stream after `Work.wait()` and before the
result finalizer. CUDA layout and copy operations started by the finalizer are
therefore ordered before later consumers on the same stream.

The handle does not support implicit consumption from multiple CUDA streams.
Code that needs that behavior must establish and own an explicit cross-stream
event protocol outside this API. Do not use the global `finalized` state as a
substitute for a per-stream dependency.

A finalizer exception is cached. Later waits re-raise the same exception
without executing the finalizer again. `owned_resources` must be a tuple and
keeps input, output, and staging objects alive until the handle itself is
released.

The opt-in two- or four-rank NCCL validation is:

```bash
VERL_RUN_ASYNC_COLLECTIVE_NCCL_TEST=1 \
VERL_ASYNC_COLLECTIVE_WORLD_SIZE=2 \
python -m pytest -q tests/utils/test_async_collective_cuda.py
```

The test deliberately rejects any world size other than two or four. It
checks same-stream asynchronous finalization, exact-once finalization, result
correctness, completion-event observation, and fail-loud use from a second
consumer stream. It does not establish multi-node behavior or performance.
