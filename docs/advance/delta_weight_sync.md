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
  measured (0.5B through 235B, 1.3–21×), not just at the large end.

This is why ``delta_sharded`` is the only delta backend we ship: an earlier full-gather variant
(diff on a rank-0 full-model snapshot) was consistently slower than ``delta_sharded`` at every
size we measured, so it was dropped in favor of the sharded design.

## Wire contract: what a rank sends to rank 0

Steady state has ONE canonical shape. Per parameter, each rank contributes a triple
``(counts[K], idx_concat int32, val_concat)`` where:

- **K is the parameter's slot count**, fixed by a static slot table identical on every
  rank. Identity params (dense DTensor, explicit blocks, unsharded) have ``K=1`` -- the
  slot is the parameter itself. Slot-enumerable converter params (``spec.hf_slots``)
  have one slot per converter output (e.g. a fused ``gate_up`` stack: ``K = E x 2``).
- The k-th slice of ``idx``/``val`` holds **final HF coordinates inside slot k**: the
  sender has already done all conversion; coordinate semantics never change after this
  point. Rank 0's whole job is slot-keyed assembly: concatenate ranks' (disjoint)
  pieces per slot, bucket, broadcast. No conversion, no rebuild, no layout knowledge.

Invariants: (1) *alignment* -- every rank enumerates the same K and slot order, with
``counts[k]=0`` for untouched slots, so the batched gather stays in lockstep; (2)
*disjointness* -- per slot, different ranks' coordinate sets never overlap (block
geometry guarantees it), so union == concatenation; (3) *bounds* -- a slot has fewer
than 2^31 elements (int32 positions), and the batched gather internally splits a
round into deterministic sub-rounds (derived from the all-gathered counts matrix, so
every rank derives the same split) whenever the largest per-rank blob would exceed
``bucket_size``. Flush triggers are count-only: they must be identical on every rank,
and byte totals are not.

Two explicit exemptions: converter params without an enumerable slot table (custom
``MOE_PARAM_HANDERS``, the Megatron shard-list contract) fall back to shard-local
payloads plus a rank-0 segmented NaN rebuild; and the dense seed for identity params
ships values only (no positions), since the first sync covers every element. The seed
for slot params is just the steady shape at full coverage.

## Design

The ``delta_sharded`` backend plugs into the standard checkpoint-engine flow (``CheckpointEngineManager`` →
``CheckpointEngineWorker``), so they work with any trainer that drives weight sync through the
checkpoint engine (including the V1 ``separate_async`` trainer).

- **Export contract**: the delta engine consumes FINAL HF-coordinate payloads; everything
  backend-specific — the weight→HF naming, the to-HF conversion, the diff and its base — lives on
  the backend side. The **seed** (first sync) streams the backend's existing full export
  ``get_per_tensor_param()`` over the values-only wire: every backend already knows how to assemble
  and convert its own full tensors (FSDP all-gather, veomni expert restack, Megatron TP/PP fusion),
  so the seed inherits all of that for free and trainer resume works by construction. After the
  seed the backend pins its shards (``prime_delta_snapshots``); every **steady** sync consumes
  ``get_per_tensor_param_delta_shard()`` — per-parameter entries ``(slots, dtype_str, counts,
  hf_idx, hf_val, gather_group)`` whose coordinates are already final HF coordinates. The engine
  only batches, gathers, buckets and ships.
- **Diff (backend-owned)**: the default strategy (shared by the FSDP and veomni backends via
  ``verl.workers.engine.utils.hf_delta_export``) byte-diffs each rank's **own shard** against its
  pinned-CPU snapshot, refreshed on every export (no rank holds a full-model snapshot). The
  comparison is bit-exact (integer view inequality), so the reconstruction is lossless by
  construction — no thresholds, no drift. A backend that already keeps the previous step's weights
  (e.g. Decoupled PPO) can diff against that checkpoint instead and skip the dedicated snapshot.
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
SGLang rollout, per-step steady-state weight sync. The ``nccl`` baseline is current main
(including the pinned-staging fix from #7005); param/optimizer offload is ON unless noted.

| model (placement) | ``delta_sharded`` | ``nccl`` (full broadcast) | speedup |
|---|---|---|---|
| Qwen2.5-7B (1+1 nodes) | **3.9-4.9 s** | 5.5-6.0 s | ~1.3x |
| Qwen2.5-32B (2+2 nodes) | **11.2-11.9 s** | 17.7-18.1 s | 1.55x |
| Qwen2.5-32B (2+2 nodes, offload off) | **6.2 s** | 14.2 s | 2.3x |
| Qwen2.5-72B (4+4 nodes, gen TP8, offload off) | **12.0-13.0 s** | 28.5-29.1 s | 2.3x |
| Qwen3-30B-A3B (veomni ep8, 1+1 nodes, 50-step medians) | **7.1 s** | 32.2 s | **4.5x** |
| Qwen3-235B-A22B (veomni ep8 x fsdp8, 8+2 nodes, gen TP16) | **11.4-14.9 s** | 246-266 s | **~21x** |

The delta sync time stays essentially flat from 32B through 235B -- the sharded sparse gather
amortizes over the larger trainer world -- while the full broadcast pays a full-model
materialization that grows linearly with parameter bytes, so the advantage widens with scale
and with MoE sparsity. The per-step changed ratio is ~1-3% of parameter bytes for dense models
(0.02-0.05% for the 235B MoE early steps) and stays there over long runs.

Correctness evidence (details in the PR):

- **200-step GRPO equivalence at 7B** (delta vs nccl, 400 syncs): reward trajectories track
  phase-for-phase, final rewards within sampling noise, zero receiver checksum failures.
- **50-step GRPO equivalence at 30B-A3B (veomni ep8)**: score trajectories rise in step
  (0.646->0.719 delta vs 0.639->0.697 nccl), per-step gap at the independent-sampling
  noise floor, zero checksum failures.
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
  ``get_per_tensor_param_shard`` export. The native mcore→HF converters are whole-param
  black boxes, outside the dim-0-separable ``to_hf_chunk`` contract; the path forward is
  rewriting them per param family as ``to_hf_chunk`` + ``hf_slots`` — the main fusions
  (interleaved qkv, gate_up concat) are row/block permutations and fit the contract, while
  TP column splits are already expressible as a ``BlockPlacement`` dim-1 offset (see #7060).
- **Quantized rollout (fp8 etc.)**: diff the quantized bytes (quantize-then-diff) so a low-precision
  rollout engine can consume deltas without a bf16 intermediate.
