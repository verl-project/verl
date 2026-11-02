# vLLM Weight Checker

## Overview

Weight Checker verifies that verl's weight synchronization path can restore model
weights on vLLM rollout replicas. Inspired by slime's snapshot/reset/compare
workflow, it runs once before training and is disabled by default.

With checking enabled, rollout independently loads the HF safetensors checkpoint
and records each replica's digests before any trainer weight transfer. It then
randomizes the weights, restores them through the actual training synchronization
path once, and compares against the independent reference. After snapshot/reset,
it calls the normal sleep and update lifecycle, including weight and KV wake-up.
Generation resumes only after comparison succeeds.

This helps investigate ineffective synchronization, missing updates, and incorrect
weight writes. Coverage depends on the vLLM worker's checksum/reset implementation
and the rollout replicas participating in the check.

## Configuration

Add these overrides to an existing vLLM training command:

```bash
actor_rollout_ref.rollout.checkpoint_engine.check_weight_update_equal=true \
actor_rollout_ref.rollout.free_cache_engine=true
```

Equivalent YAML:

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    free_cache_engine: true
    checkpoint_engine:
      check_weight_update_equal: true
```

This is a configuration fragment, not a standalone training configuration. Keep
the model, training engine, resources, and synchronization backend settings from
your existing job.

`free_cache_engine=true` exercises normal rollout sleep/wake, including level 2
where selected by the engine. Only the early initialization sleep is deferred
until the reference has been recorded. Later training sleep/wake calls are unchanged.
With `false`, verification still runs but does not exercise memory release.
With checking disabled, dummy remains the default
loader. With checking enabled, the effective loader becomes `safetensors`;
explicit incompatible loaders are rejected. Full unquantized BF16 weights are
required; LoRA (including merged adapters) and speculative decoding are excluded.
Teacher and reward-model replicas do not enable the actor weight diagnostic.

On restored runs (including restored step-zero checkpoints), the trainer skips verification
and performs normal synchronization. The feature does not automatically check
weights after every training update.

## Verification Workflow

1. Independently load rollout weights from the same HF safetensors checkpoint
   used by the trainer, with the same BF16 dtype and compatible model settings.
2. Pause rollout and run `prepare` to validate the unsynchronized reference without changing
   weight or KV memory allocations.
3. Run `snapshot` to store each worker's weight digests on the server.
4. Run `reset` to randomize the weights covered by the vLLM checker.
5. Call `sleep_replicas()`, then perform the first full trainer synchronization
   with the normal weight/KV memory lifecycle. Keep generation paused.
6. Run `compare` to compare digest values and key sets, detecting changed,
   missing, or newly added entries.
7. After every replica passes, run `finish` to invalidate prefix-cache contents
   and clear the digest baseline, then resume generation.

The baseline is a server-side digest dictionary and survives level-2 sleep.
Weight synchronization must restore every checksum-covered tensor. Pausing and
cache invalidation prevent reuse of stale KV values after weight changes.

`snapshot` stores a digest dictionary rather than a full model copy on the verl
server. Digest keys include worker parallel ranks. Duplicate keys raise an error
instead of silently overwriting an entry.

Responses of `None` from headless servers are filtered out. A replica without a
valid baseline or comparison result causes verification to fail.

After a failed check, generation remains paused because weights may have been
randomized. Resolve the failure and restart the job, or explicitly restore all
weights before resuming generation.

## Verification Logs

Seeing `check_weight_update_equal=true` in startup overrides only confirms that
the configuration was supplied. An actual check produces these messages:

```text
Weight update check: loading independent safetensors reference
Weight update check: generation paused; recording reference before sleep
Weight update check: snapshotting N rollout replica(s)
Weight update check: independent reference recorded, M weight shard(s)
Weight update check: weights reset, syncing through the real update path
Weight update check: comparing against independent reference
Weight update check passed: N replica(s), M weight shard(s) matched.
```

The final `Weight update check passed` message confirms successful completion.
`M` counts aggregated digest keys, not GPUs or individual parameter elements.

A resumed run that skips verification produces a message beginning with:

```text
Skipping weight update check on a resumed run
```

To filter the logs:

```bash
grep -nE 'Weight update check|weight update check|Skipping weight update check' run.log
```

## Compatibility and Limitations

| Scenario | Behavior |
| --- | --- |
| Internal vLLM DP=1 with newer APIs | Prefer the newer checksum/reset APIs. |
| Internal vLLM DP=1 with older APIs | Use `compute_weight_checksums()` or dispatch to workers through `collective_rpc`. |
| Internal vLLM DP>1 | Require `compute_weight_checksums_all()` and `reset_weights()`. |
| No dedicated KV release API | Supported: the diagnostic does not release KV memory. |
| `free_cache_engine=true` | Exercise normal sleep/wake after reference capture; later training behavior is unchanged. |
| `free_cache_engine=false` | Verify the transfer without exercising sleep/wake memory release. |
| Precision and model | BF16, no quantization, no LoRA or speculative decoding. |
| `delta_sharded` backend | Reject the diagnostic: unchanged deltas cannot restore randomized weights. |
| Adapter-only LoRA synchronization | Reject the diagnostic: adapter updates cannot restore randomized base weights. |

DP here means `rollout.data_parallel_size` within a vLLM instance, rather than
the training FSDP world size or the number of rollout replicas.

The legacy RPC path still requires vLLM workers to implement
`compute_weight_checksums` and `reset_weights`. The compatibility layer does not
add worker-side checking to vLLM versions that lack it.

The baseline comes from rollout's independent HF load. The diagnostic verifies
whether the first trainer transfer reproduces that reference on each replica:

- It does not directly compare initial weights between replicas.
- It compares engine weights against the independent runtime reference, not
  raw checkpoint-file hashes. It assumes the HF loader itself is correct.
- A deterministic error shared by both loading paths can still escape detection.
- Digest comparison is exact, with no quantization error tolerance.
  Nondeterministic quantization or conversion can cause a mismatch.

## Memory and Troubleshooting

The diagnostic adds independent HF loading, two digest computations, and one weight
randomization; it uses one trainer synchronization. Digest computation may copy tensors to
the CPU.

The KV pool is restored before comparison by the normal update path. Reserve enough device memory
for reset temporaries, training weights, weight export, FSDP all-gather, and
synchronization buckets in addition to the rollout weights and KV pool.
Snapshot and reset happen before sleep and still need temporary memory. If this exceeds
available memory, reduce the rollout memory budget or synchronization bucket size,
or enable suitable training-side offload settings.

- For OOM during FSDP initialization, before checker messages appear, inspect
  other GPU processes and initialization memory peaks first.
- For OOM during reset or synchronization, account for the resident KV
  pool and peak memory used by reset, weight export, and synchronization buckets.
- For `DP>1 weight checking requires ...`, use a vLLM version providing the
  required all-DP APIs. A single-shard RPC is insufficient for this configuration.
- For `Duplicate weight checksum keys`, inspect worker digest rank prefixes.
- For `weight update check failed`, inspect the reported mismatch keys from the
  first failed result and the corresponding weight loading mappings.

## Tests and Code References

From the repository root, in an environment with the project test dependencies:

```bash
pytest -q tests/checkpoint_engine/test_weight_checker.py
```

The tests use mocks to cover digest merging/comparison, API compatibility, DP
restrictions, verification ordering, failure handling, normal naive worker calls,
and distributed KV restoration without prematurely resuming generation.
They require no GPU and do not launch a
real vLLM engine. They do not replace memory and synchronization correctness
validation in an actual training environment.

| File | Responsibility |
| --- | --- |
| [base.py](../verl/checkpoint_engine/base.py) | `verify_update_path()` and synchronization lifecycle. |
| [engine_workers.py](../verl/workers/engine_workers.py) | Run normal weight/KV wake-up and transfer during verification. |
| [vllm_async_server.py](../verl/workers/rollout/vllm_rollout/vllm_async_server.py) | Handle prepare/snapshot/reset/compare/finish actions. |
| [weight_checksum.py](../verl/utils/weight_checksum.py) | Adapt vLLM APIs and merge/compare digests. |
| [test_weight_checker.py](../tests/checkpoint_engine/test_weight_checker.py) | Consolidated regression tests. |
