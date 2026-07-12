# Delta Weight Sync

Last updated: 07/10/2026.

## Motivation

In a disaggregated setup (``hybrid_engine=False``) the trainer must broadcast its updated weights to the
rollout engine after every step. By default this is a full-weight broadcast whose cost grows with model
size. Because RL updates are highly sparse — under typical learning rates over 99% of BF16 weight bytes
are unchanged step-over-step — you can instead broadcast only the parameters that changed (a *delta*),
cutting the weight-sync traffic to the sparsity ratio while staying lossless (bit-exact; a per-flush
checksum is verified on the receiver).

When to use: disaggregated training where the trainer↔rollout link is the bottleneck (commodity network /
cross-node / large models). On a fast intra-node NVLink link with a small model the full broadcast is
already cheap, so delta mainly pays off as the model size and the network distance grow.

## Design

The delta backends plug into the standard checkpoint-engine flow (``CheckpointEngineManager`` →
``CheckpointEngineWorker``), so they work with any trainer that drives weight sync through the
checkpoint engine (including the V1 ``separate_async`` trainer).

- **Diff**: the trainer byte-diffs each parameter against a pinned-CPU snapshot of the previous sync.
  The comparison is bit-exact (integer view inequality), so the reconstruction is lossless by
  construction — no thresholds, no drift.
- **Encoding**: changed elements are shipped as a shared ``(positions, values)`` payload plus a
  per-parameter manifest. ``encoding`` selects the position encoding: ``indices`` (int32 absolute
  positions, lowest compute) or ``deltas`` (uint16 gap deltas, smaller on the wire).
- **Transport**: the sparse payload is broadcast over the existing NCCL collective group in
  bucket-sized flushes (streamed: each flush is sent and freed as it is produced, so sender peak
  memory stays ~2 buckets regardless of model size).
- **Apply**: each rollout worker hands its local copy of the sparse payload to its colocated SGLang
  TP worker via same-GPU ``update_weights_from_tensor`` IPC, where a verl-shipped loader —
  registered automatically through SGLang's stock ``--custom-weight-loader`` hook, so **no SGLang
  fork or patch is needed** — verifies the flush checksum (fail loud), densifies each parameter's
  delta into a NaN-masked tensor, and overwrites only the changed positions *in place* on the live
  weights. No full-model mirror is staged anywhere on the rollout side: receiver peak memory is one
  bucket plus one decode chunk, independent of model size.
- **Seeding**: the first sync is an explicit **dense** pass — the raw weights stream through the same
  bucketed wire with no positions attached (values only), populating the trainer-side snapshot as they
  go — so a dummy-initialized rollout gets a correct base without any sparse-encoding overhead.
  Subsequent syncs are sparse.

## Backends

### ``delta``

Diffs on rank 0 after the ordinary full-tensor gather; rank 0 holds a full-model pinned-CPU snapshot.

```shell
    actor_rollout_ref.rollout.checkpoint_engine.backend=delta \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta.encoding=indices
```

### ``delta_sharded`` (sharded snapshot)

The ``delta`` backend above still ``full_tensor()``-gathers every parameter to rank 0 before diffing.
The ``delta_sharded`` backend pushes the diff *below* the all-gather: each actor rank pins a snapshot of
only **its** FSDP shard, byte-diffs the shard locally, and gathers just the changed ``(position, value)``
pairs to rank 0 (via the engine's ``get_per_tensor_param_shard()`` export). So the gather volume drops
from the full parameter to the sparsity ratio (~1–3%), and no rank needs a full-model snapshot — the
memory and the gather traffic both shard with the world size.

```shell
    actor_rollout_ref.rollout.checkpoint_engine.backend=delta_sharded \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.encoding=indices
```

The assembled delta is **bit-identical** to what ``delta`` produces, so the wire format, the per-flush
checksum, and the rollout-side receiver are all unchanged. Each rank computes its shard's absolute
position in the full flattened parameter purely locally (from the DTensor spec, no extra collective).

**Supported training engines**: the shard export requires ``Shard(0)`` DTensor parameters, which both
FSDP versions provide:

- **FSDP2** (``fully_shard``, ``actor.strategy=fsdp2``): native DTensor params; the export never stages
  the whole shard on the GPU (``state_dict()`` is reference-only, shards move lazily per parameter).
- **FSDP1** (``actor.strategy=fsdp``, the default): verl configures ``SHARDED_STATE_DICT``, whose export
  also emits per-rank ``Shard(0)`` DTensors. FSDP1's state-dict export runs through the unshard
  machinery, so the whole-shard GPU staging round trip is kept for it (it is skipped for FSDP2).
  Single-GPU FSDP1 uses ``FULL_STATE_DICT`` (plain tensors) and degrades to the replicated/rank-0 path —
  still correct, just not shard-parallel.

Other shard dimensions than ``Shard(0)`` are not supported and raise.

> **Config note**: the training engine reads the **top-level** ``actor_rollout_ref.actor.strategy``;
> setting only ``actor.fsdp_config.strategy`` does *not* select FSDP2.

## Usage

A runnable example is ``verl/experimental/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_sglang_delta_2_6.sh`` —
the SGLang 2+6 disaggregated GRPO recipe with ``backend=delta``.

Current scope: disaggregated (``hybrid_engine=False``) + SGLang rollout in BF16, FSDP1/FSDP2 training engines.
Selecting a delta backend with any other rollout engine raises ``NotImplementedError`` at worker startup;
a per-backend apply interface (vllm/trt-llm plugins) is planned.

## Roadmap

Planned extensions, in design order:

- **Megatron sharded delta**: diff each rank's mcore shard locally and reuse the native
  megatron→HF bridge on rank 0, extending ``delta_sharded`` beyond FSDP.
- **Quantized rollout (fp8 etc.)**: diff the quantized bytes (quantize-then-diff) so a low-precision
  rollout engine can consume deltas without a bf16 intermediate.
