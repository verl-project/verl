# Native rank-level phase capture

Last updated: 09/05/2026.

The phase-sweep benchmark can emit an opt-in, fresh JSONL artifact per rank:

```bash
torchrun --standalone --nproc-per-node=4 scripts/benchmark_collective_phase_sweep.py \
  --device cuda --backend nccl --group-layout auto \
  --comm-a all-to-all --comm-b all-reduce \
  --message-bytes-a 4MiB --message-bytes-b 4MiB \
  --policies concurrent serialized --warmup 2 --iters 10 \
  --output-json capture/summary.json --trace-jsonl 'capture/raw-{rank}.jsonl'
```

Raw schema version 3 is separate from the unchanged summary schema version 2.
Every rank receives the same fresh process-launch/run UUID. Missing `{rank}`
automatically adds a rank suffix. All ranks preflight resolved path uniqueness,
summary collision, writability and exclusive creation before any trial. Existing
raw files are never appended or overwritten. An initialization failure can leave
closed empty files as failure evidence; choose a fresh output directory to retry.

Each collective retains its actual process-group name **and** members, its
communicator sequence (including warmup), message bytes, stream, API launch and
return, host-observed physical completion, and first payload-validation consumer.
The first-consumer timestamp is absent when `--no-validate` is selected; the
writer does not invent a deadline. CUDA start/end timestamps are event brackets;
Gloo records contain **null GPU timestamps**, never repackaged host durations.
Cross-rank clock uncertainty remains unknown until separately calibrated.

`buffer_reuse_acquire_timestamp_ns` and `buffer_reuse_release_timestamp_ns`
describe the benchmark's **persistent-buffer transfer lease**: from use by this
transfer to physical completion and optional correctness consumption, before
the next reset is permitted. These are not allocator malloc/free observations
and do not measure total GPU memory residency. JSON serialization occurs after
the measured transfer/consumer boundaries and is explicitly opt-in; capture can
still perturb the experiment and must be controlled in independent confirmation.

Each record names `sample_phase` (`warmup` or `measurement`) and `policy_id`.
Offsets have distinct IDs, e.g. `offset/-1000us`; they must not be merged into a
single scoring cell. Pair IDs use actual trial ordinals, not reconstructed
percentile rows. One sweep is one process invocation, not multiple independent
confirmation runs.

The matching InfraSWE importer accepts schema 3 JSONL with `--framework verl`
and an explicit `--policy-id concurrent`. It preserves observed leases and
refuses GPU certification for Gloo captures. Schema 2 summaries still cannot be
used to reconstruct per-rank records. Two/four-rank Gloo smoke tests exercise the
real collective callbacks, all sequence IDs, warmup separation and lifecycle
timestamps; they do not certify CUDA/NCCL performance.
