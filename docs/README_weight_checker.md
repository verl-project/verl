# vLLM Weight Checker

## Overview

Weight Checker verifies that verl's weight synchronization path can restore model
weights on vLLM rollout replicas. Inspired by slime's snapshot/reset/compare
workflow, it runs once before training and is disabled by default.

The diagnostic first synchronizes weights normally and records each replica's
weight digests. It then randomizes the weights, restores them through the actual
training synchronization path, and compares the resulting digests against the
baseline. Generation stays paused during reset and the second transfer while
the KV cache memory pool remains allocated. Generation resumes only on success.

This helps investigate ineffective synchronization, missing updates, and incorrect
weight writes. Coverage depends on the vLLM worker's checksum/reset implementation
and the rollout replicas participating in the check.

## Configuration

Add these overrides to an existing vLLM training command:

```bash
actor_rollout_ref.rollout.checkpoint_engine.check_weight_update_equal=true \
actor_rollout_ref.rollout.free_cache_engine=false
```

Equivalent YAML:

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    free_cache_engine: false
    checkpoint_engine:
      check_weight_update_equal: true
```

This is a configuration fragment, not a standalone training configuration. Keep
the model, training engine, resources, and synchronization backend settings from
your existing job.

`free_cache_engine=false` disables the normal rollout sleep/wake memory lifecycle
for a pause/resume-only job. The checker also works with `true`: the initial
bootstrap synchronization follows the normal lifecycle, then the diagnostic
window retains the resident weights and KV pool without further sleep/wake calls.

On resumed runs with `global_steps > 0`, the current trainer skips verification
and performs normal synchronization. The feature does not automatically check
weights after every training update.

## Verification Workflow

1. Pause rollout requests and perform the initial normal weight synchronization
   to establish valid model weights.
2. Pause rollout and run `prepare` to validate checker support without changing
   weight or KV memory allocations.
3. Run `snapshot` to store each worker's weight digests on the server.
4. Run `reset` to randomize the weights covered by the vLLM checker.
5. Perform a second full synchronization, retaining KV allocations and keeping
   generation paused.
6. Run `compare` to compare digest values and key sets, detecting changed,
   missing, or newly added entries.
7. After every replica passes, run `finish` to invalidate prefix-cache contents
   and clear the digest baseline, then resume generation.

Retaining the KV pool does not mean reusing old KV values after a weight update.
Pausing aborts in-flight work, and cache invalidation prevents stale values from
being reused. These operations retain the pool's device memory allocation.

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
Weight update check: generation paused; retaining KV cache memory
Weight update check: snapshotting N rollout replica(s)
Weight update check: baseline recorded, M weight shard(s)
Weight update check: weights reset, syncing through the real update path
Weight update check: comparing against the baseline
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
| `free_cache_engine=false` | Supported: keep rollout memory resident throughout the job. |
| `delta_sharded` backend | Reject the diagnostic: unchanged deltas cannot restore randomized weights. |
| Adapter-only LoRA synchronization | Reject the diagnostic: adapter updates cannot restore randomized base weights. |

DP here means `rollout.data_parallel_size` within a vLLM instance, rather than
the training FSDP world size or the number of rollout replicas.

The legacy RPC path still requires vLLM workers to implement
`compute_weight_checksums` and `reset_weights`. The compatibility layer does not
add worker-side checking to vLLM versions that lack it.

The baseline comes from the first synchronization. The diagnostic verifies
whether each replica can restore its own baseline through the second transfer:

- It does not directly compare initial weights between replicas.
- It does not directly compare synchronized weights against the original checkpoint.
- Passing does not prove that naming, conversion, or quantization is semantically
  correct; both synchronizations can reproduce the same error.
- Digest comparison is exact, with no quantization error tolerance.
  Nondeterministic quantization or conversion can cause a mismatch.

## Memory and Troubleshooting

The diagnostic adds one synchronization, two digest computations, and one weight
randomization, increasing startup time. Digest computation may copy tensors to
the CPU.

The KV pool remains resident during verification. Reserve enough device memory
for reset temporaries, training weights, weight export, FSDP all-gather, and
synchronization buckets in addition to the rollout weights and KV pool.
Pause/resume alone does not reduce device memory consumption. If this exceeds
available memory, reduce the rollout memory budget or synchronization bucket size,
or enable suitable training-side offload settings.

- For OOM during FSDP initialization, before checker messages appear, inspect
  other GPU processes and initialization memory peaks first.
- For OOM during reset or the second synchronization, account for the resident KV
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
restrictions, verification ordering, failure handling, and the KV restoration
control used by naive synchronization, and the absence of KV lifecycle calls
during the diagnostic transfer. They require no GPU and do not launch a
real vLLM engine. They do not replace memory and synchronization correctness
validation in an actual training environment.

| File | Responsibility |
| --- | --- |
| [base.py](../verl/checkpoint_engine/base.py) | `verify_update_path()` and synchronization lifecycle. |
| [engine_workers.py](../verl/workers/engine_workers.py) | Skip weight/KV wake-up calls during the paused diagnostic transfer. |
| [vllm_async_server.py](../verl/workers/rollout/vllm_rollout/vllm_async_server.py) | Handle prepare/snapshot/reset/compare/finish actions. |
| [weight_checksum.py](../verl/utils/weight_checksum.py) | Adapt vLLM APIs and merge/compare digests. |
| [test_weight_checker.py](../tests/checkpoint_engine/test_weight_checker.py) | Consolidated regression tests. |
