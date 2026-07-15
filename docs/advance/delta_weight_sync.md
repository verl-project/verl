# Delta Weight Sync

Last updated: 07/10/2026.

## Motivation

In a disaggregated setup (``hybrid_engine=False``) the trainer must broadcast its updated weights to the
rollout engine after every step. By default this is a full-weight broadcast whose cost grows with model
size. Because RL updates are highly sparse — under typical learning rates over 99% of BF16 weight bytes
are unchanged step-over-step — you can instead broadcast only the parameters that changed (a *delta*),
cutting the weight-sync traffic to the sparsity ratio while staying lossless (bit-exact; a per-flush
checksum is verified on the receiver).

When to use: disaggregated training with a trainer↔rollout link. Two effects stack here, and they
pay off differently:

- **Sparse wire (the "delta" part)**: only ~1–3% of parameter bytes change per step, so the
  broadcast payload shrinks accordingly. This effect grows with model size and network distance —
  on a fast intra-node link with a small model, a full broadcast is already cheap.
- **Shard-local diff + sparse gather (the "sharded" part)**: no rank ever materializes full
  tensors or a full-model snapshot, and the gather moves only changed elements. This removes the
  full-tensor all-gather and rank-0 staging costs that the plain ``nccl`` engine pays *regardless
  of network speed* — which is why ``delta_sharded`` beat the full broadcast at every size we
  measured (0.5B through 72B, 1.3–3.1×), not just at the large end.

This is why ``delta_sharded`` is the only delta backend we ship: an earlier full-gather variant
(diff on a rank-0 full-model snapshot) was consistently slower than ``delta_sharded`` at every
size we measured, so it was dropped in favor of the sharded design.

## Design

The ``delta_sharded`` backend plugs into the standard checkpoint-engine flow (``CheckpointEngineManager`` →
``CheckpointEngineWorker``), so they work with any trainer that drives weight sync through the
checkpoint engine (including the V1 ``separate_async`` trainer).

- **Export contract**: the trainer's ``get_per_tensor_param_shard()`` yields
  ``(name, local_shard, ShardSpec)`` per local parameter — the spec (see
  :mod:`verl.workers.engine.spec`) describes the shard's placement declaratively (DeviceMesh +
  Placements), and the engine derives the flat offset, gather group, and contributing rank itself.
  All layout knowledge stays on the trainer side; the engine is trainer-agnostic.
- **Diff**: each rank byte-diffs **its own shard** against a pinned-CPU snapshot of that shard from
  the previous sync (no rank holds a full-model snapshot). The comparison is bit-exact (integer
  view inequality), so the reconstruction is lossless by construction — no thresholds, no drift.
- **Sparse gather + encoding**: only the changed ``(position, value)`` pairs are gathered to rank 0
  (batched, variable-length), translated to full-tensor coordinates, and packed as a shared
  ``(positions, values)`` payload plus a per-parameter manifest (``indices`` encoding: int32
  absolute positions).
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

## Backend

### ``delta_sharded`` (sharded snapshot)

``delta_sharded`` pushes the diff *below* the all-gather: each actor rank pins a snapshot of
only **its** FSDP shard, byte-diffs the shard locally, and gathers just the changed ``(position, value)``
pairs to rank 0 (via the engine's ``get_per_tensor_param_shard()`` export). So the gather volume drops
from the full parameter to the sparsity ratio (~1–3%), and no rank needs a full-model snapshot — the
memory and the gather traffic both shard with the world size.

```shell
    actor_rollout_ref.rollout.checkpoint_engine.backend=delta_sharded \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.encoding=indices
```

The assembled delta is **bit-identical** to full-gather-then-diff, so the wire format, the per-flush
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

## Measured results

All numbers: H100 nodes, GSM8K GRPO, verl V1 ``separate_async`` (disaggregated trainer/rollout),
FSDP2 + param/optimizer offload, SGLang rollout, per-step steady-state weight sync.

| model (placement) | ``delta_sharded`` | ``nccl`` (full broadcast) | speedup | saved / step |
|---|---|---|---|---|
| Qwen2.5-7B (1+1 nodes, sustained over 200 steps) | **3.8 s** | 9.1 s | 2.4x | 5.2 s |
| Qwen2.5-32B (2+2 nodes) | **12.5 s** | 23.2 s | 1.9x | 10.7 s |
| Qwen2.5-72B (4+4 nodes, TP8) | **12.0 s** | 36.9 s | **3.1x** | **24.9 s** |

The delta sync time stays essentially flat from 32B to 72B -- the sharded sparse gather amortizes
over the larger trainer world -- while the full broadcast grows linearly with parameter bytes, so
the advantage widens with scale. The per-step changed ratio is stable at ~1-3% of parameter bytes
across sizes and stays there over long runs.

Correctness evidence (details in the PR):

- **200-step GRPO equivalence at 7B** (delta vs nccl, 400 syncs): reward trajectories track
  phase-for-phase, final rewards within sampling noise, zero receiver checksum failures.
- **Bit-exact round-trip**: perturb -> apply as delta -> revert -> apply as delta reproduces
  greedy generations byte-identically on every prompt.

## Usage

A runnable example is ``verl/experimental/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_sglang_delta_sharded_2_6.sh`` —
the SGLang 2+6 disaggregated GRPO recipe with ``backend=delta_sharded``.

Current scope: disaggregated (``hybrid_engine=False``) + SGLang rollout in BF16, FSDP1/FSDP2 training engines.
Selecting a delta backend with any other rollout engine raises ``NotImplementedError`` at worker startup;
a per-backend apply interface (vllm/trt-llm plugins) is planned.

## Roadmap

Planned extensions, in design order:

- **Megatron-core trainers**: the same ``delta_sharded`` backend via a Megatron
  ``get_per_tensor_param_shard`` export whose spec carries the native mcore→HF conversion as a
  pure-permutation ``to_hf`` closure (implemented and validated in a stacked follow-up PR).
- **Quantized rollout (fp8 etc.)**: diff the quantized bytes (quantize-then-diff) so a low-precision
  rollout engine can consume deltas without a bf16 intermediate.
