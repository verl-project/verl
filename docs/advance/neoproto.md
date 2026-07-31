# NeoProto data plane

Last updated: 07/31/2026.

NeoProto is an experimental ref/index data plane for the V0
`RayPPOTrainer`. It keeps batch payloads in a storage engine while the
controller primarily manipulates schemas, references, and index views.

NeoProto is disabled by default:

```yaml
trainer:
  use_neoproto: false
```

Set `trainer.use_neoproto=true` to enable it. The classic `DataProto` path is
unchanged when the option is false.

## Scope

This initial integration covers:

- the synchronous V0 `RayPPOTrainer`;
- Ray object-store backed storage;
- DataProto-compatible selection, repeat, chunk, reorder, concat, union, and
  padding operations;
- actor, reference-policy, critic, rollout, reward-loop, and agent-loop data
  flow;
- worker-side conversion to TensorDict before existing compute engines run.

It does not add NeoProto support to the V1/fully asynchronous trainer, and it
does not change actor, critic, rollout, reward, or advantage algorithms.

## Architecture

The driver creates a NeoProto-backed DataProto view at dataloader ingress.
Tensor and non-tensor fields are represented by a schema and a ref table. Batch
transformations update references and index views without eagerly copying the
payload.

Before Ray sends a chunk to a worker, the dispatch layer attaches the
rank-specific ref table. The worker bridge then materializes the fields required
by the compute entry point, converts them to TensorDict, and invokes the
existing engine implementation. Worker outputs are converted back into
NeoProto-backed DataProto objects.

```text
dataloader
    |
    v
NeoProto-backed DataProto
    |
    +-- driver operations: ref/schema/index only where possible
    |
    v
rank-specific Ray dispatch
    |
    v
worker materialize -> TensorDict -> existing compute engine
    |
    v
NeoProto-backed result
```

## Correctness semantics

NeoProto follows the public DataProto behavior for duplicate keys, indexing, and
dtype conversion:

- boolean masks select rows in the current logical batch order;
- boolean mask length must match the logical batch length;
- `union` rejects conflicting duplicate payloads instead of silently keeping
  one side;
- inference outputs that the classic V0 path exposes as FP32 are also FP32 on
  the NeoProto path;
- strict mode rejects an unexpected non-Neo worker result instead of silently
  falling back to an empty payload.

GAE remains on the driver in this initial integration. This preserves classic
global-batch advantage whitening. Computing GAE independently on data-parallel
chunks would use a different whitening domain and is therefore not treated as a
transparent data-plane optimization.

Training metrics are also computed with the classic global reductions.
Pre-aggregated per-rank masked means or explained-variance values cannot be
averaged without changing their semantics when ranks contain different numbers
of valid tokens.

## Strict verification mode

`trainer.neoproto_strict_mode=true` is intended for correctness tests. It
requires ref-table dispatch, rejects forced full-materialization mode, and
fails if a worker output unexpectedly bypasses the NeoProto path.

`trainer.correctness_dump_dir` optionally writes per-step semantic snapshots for
the deterministic DataProto-versus-NeoProto E2E test. Leave it `null` during
normal training.

```yaml
trainer:
  use_neoproto: true
  neoproto_strict_mode: true
  correctness_dump_dir: null
```

## Validation

The focused unit tests cover the container and transfer plane:

```bash
pytest -q \
  tests/experimental/neoproto/test_neoproto.py \
  tests/experimental/neoproto/test_neo_plane.py
```

The deterministic E2E runs the existing FSDP + vLLM function-reward case twice,
once with DataProto and once with NeoProto:

```bash
bash tests/special_e2e/ppo_trainer/run_neoproto_correctness.sh
```

The E2E is fail-closed. It verifies successful and complete training, finite
semantic metrics, explicit NeoProto/strict-mode markers, Neo materialization
activity, decoded rollout equivalence, per-step snapshots, and native
checkpoint contents.

### Two-node correctness-run performance

The deterministic correctness run was also exercised on two nodes with four
NVIDIA H20 GPUs per node (eight GPUs total). The run used Qwen3-8B, FSDP actor
and critic workers, asynchronous vLLM rollout with tensor parallel size 2, GAE,
16 prompts per step, four responses per prompt, and a maximum response length
of 512. Both paths used the same deterministic settings and completed two
training steps; step 2 includes native checkpoint writing.

| Data path | Step 1 (s) | Step 2 (s) | Two-step total (s) | Aggregate throughput (tokens/s/GPU) |
| --- | ---: | ---: | ---: | ---: |
| DataProto | 68.098 | 72.054 | 140.153 | 68.044 |
| NeoProto | 65.903 | 71.561 | 137.464 | 69.375 |

For this run, NeoProto reduced the two-step wall time by 1.92% and increased
aggregate throughput by 1.96%. Its measured materialization time was about
45 ms per step across 30 materialization calls. The equivalence checker also
confirmed identical semantic metrics, rollouts, per-step snapshots, and all
eight ranks of the native checkpoints.

This is a two-step correctness smoke benchmark, not a statistically robust
throughput benchmark. Full determinism, correctness dumps, rollout dumps, and
checkpoint writing are enabled, so the result should be interpreted as evidence
that the tested multi-node path does not introduce a material regression, not
as a general performance claim.

## Limitations

- Only V0 synchronous PPO is supported.
- The initial backend is the Ray object store.
- Worker-distributed GAE is intentionally excluded because per-chunk whitening
  is not equivalent to global whitening.
- Worker-side scalar metric aggregation is intentionally excluded until it can
  reproduce global masked reductions.
- NeoProto is experimental and remains opt-in.
