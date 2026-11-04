# Sharded NIXL Weight Sync

Last updated: 10/22/2026.

RFC for the ``sharded_nixl`` checkpoint engine: trainer-to-rollout weight sync in which every trainer rank
writes the shards it holds straight into the rollout workers' buffers over NIXL, with no trainer-side
all-gather. Code: ``verl/checkpoint_engine/sharded_nixl_checkpoint_engine.py`` (protocol),
``sharded_nixl_layout.py`` (planning) and ``verl/workers/rollout/vllm_rollout/rank_local_bake.py`` (vLLM
TP slices).

## Motivation

In a disaggregated setup (``hybrid_engine=False``) every step ends with a full weight sync. The ``nccl``
and ``nixl`` engines all-gather full tensors on the trainer and send the whole model from trainer rank 0
to every rollout worker. Two costs follow:

- **One sender.** The sync is bounded by rank 0's gather and link, so a larger trainer does not make it
  faster: at Qwen3-235B-A22B both engines take ~20 s whether the trainer has 16 or 32 GPUs.
- **Everything to everyone.** Every rollout worker receives every parameter, although with rollout EP it
  loads only its experts, with PP only its layers, and with TP only a slice of each tensor.

The weights are already sharded on the trainer; ``sharded_nixl`` keeps them sharded on the wire.

## Design

``sharded_nixl`` is a ``CheckpointEngine`` backend, so it plugs into the same
``CheckpointEngineManager`` -> ``CheckpointEngineWorker`` flow as the other engines.

**Export.** The engine takes the training engine in ``send_weights(engine, global_steps)`` and exports
each rank's local shards: ``get_per_tensor_param_shard()`` for FSDP (a ``Shard(0)`` or replicated shard
is one byte interval of the flattened tensor) and ``get_per_tensor_param_hf_shard()`` for Megatron (local
shards in final HF coordinates, reusing the ``delta_sharded`` Megatron-Bridge probe). Nothing is sent
twice: TP-sharded Megatron parameters come from data-parallel rank 0 only, and replicated experts
round-robin over the expert-data-parallel group.

**Ownership.** Before the first sync every rollout worker reports a ``RolloutLayout`` through
``BaseRollout.get_weight_ownership()``: TP/PP/EP ranks, layer range, resident experts and accepted wire
formats. The engine drops what a worker does not load, and expands fused MoE tensors into the per-expert
HF names vLLM loads. A rollout that returns ``None`` receives everything.

**Static layout.** Every trainer rank exports the same ``(name, shape, dtype)`` sequence, so all ranks
compute the same bucket layout without communicating. The rollout side caches the layout and each trainer
rank's write tables by digest, so after the first sync a sync exchanges only small headers.

**Transport.** Per sync (S = sync id, NB staging buffers per worker):

```text
rollout worker g -> every trainer rank: hello{S, rollout layout, cached digests}     (ZMQ)
trainer rank 0   -> every worker:       sync{S, layout digest, num_buckets}           (ZMQ)
for bucket k:
    trainer rank r: wait ACK(S, k-NB) from every worker; pack its slices into buffer k % NB;
                    NIXL WRITE them into every worker's buffer k % NB with notification SNX{S, k, r, hashes}
    worker g:       wait SNX from every trainer rank; check hashes; load bucket k; send ACK{S, k}
```

Writes are one-sided, so there is no NCCL group between trainer and rollout and every trainer rank sends
in parallel. Staging buffers live in GPU memory (GPUDirect RDMA) or pinned host memory.

**Rank-local TP slices.** When every rollout worker advertises ``rank_local_named_tensors`` with a table
of the slices it keeps, the engine plans a second layout over those local shapes, and each TP rank
receives only its slices; pieces that several workers need are packed once. The slices come from vLLM
itself: a dry run of ``model.load_weights`` feeds each checkpoint tensor as a storage-less stand-in that
records ``narrow()`` calls and catches the final ``copy_``. A name copied as one strict sub-box of one
parameter gets a slice; anything else is sent whole and goes through ``load_weights``. Received slices
are copied straight into their parameter views.

**Integrity and failure.** Every written slice carries a 64-bit hash, checked on the rollout side before
any weight is applied, and sync ids must match. A sync that fails part-way leaves a mix of old and new
weights, so ``CheckpointEngineManager`` keeps ``resume_generation_replicas`` raising until a later
``update_weights`` completes; every sync is dense, so that retry repairs every weight. Before the retry,
``prepare`` cancels the failed sync's leftover send or receive and drops its stale notifications.

**API.** Two opt-in ``CheckpointEngine`` class attributes, both ``False`` by default:
``send_weights_takes_engine`` (``send_weights`` gets the training engine instead of an all-gathered
iterator; ``delta_sharded`` now uses it too) and ``needs_rollout_layout`` (the worker's ``RolloutLayout``
goes to ``set_rollout_layout`` before the first receive). ``BaseRollout.get_weight_ownership()`` returns
``None`` by default; the vLLM ``ServerAdapter`` implements it.

## Alternatives considered

- **NCCL resharding (#7432, #7552, #8087, #8088).** The same goal, each worker receiving only its
  destination shards, over NCCL collectives between trainer and rollout. #8087 uses the same
  ``rank_local_named_tensors`` wire format name, so the two vLLM receive paths could be shared.
- **Mooncake P2P (#7108).** Per-rank RDMA writes from Megatron to SGLang over Mooncake. This RFC targets
  vLLM over NIXL and also supports FSDP trainers.
- **Sparse deltas (``delta_sharded``).** Orthogonal: it shrinks what is sent, not who sends it, and still
  funnels through rank 0. Combining the two is future work.

## Measured results

B300 nodes, 400 Gb/s RoCE, GPUDirect RDMA, NIXL 1.4.1 (UCX), vLLM 0.24.0, 512 MB buckets.

| Setup | ``nixl`` | ``nccl`` | ``sharded_nixl`` |
|---|---|---|---|
| Qwen3-235B-A22B, Megatron TP=4/PP=2/CP=2/EP=8 (16 GPUs) to vLLM TP=16/EP=16 (16 GPUs) | 21.1 s | 21.7 s | **6.35 s** |
| Same, trainer TP=4/PP=4/CP=2/EP=8 (32 GPUs) | 19.8 s | 21.5 s | **6.36 s** |
| GRPO, Qwen3-30B-A3B, Megatron TP=4/PP=2/EP=4 to 2 x vLLM TP=4/EP=4 | 9.3 s | 9.4 s | **7.5 s** |

The 235B rows come from a standalone benchmark that checks vLLM's greedy output against reference tokens
after every sync; they were measured before slice hashing became mandatory, with it off. The GRPO row is the median ``update_weights`` time over 6 steps of
``tests/special_e2e/run_v1_separate_async.sh`` on GSM8K, where rollout-vs-trainer log-prob agreement
matched ``nccl`` and ``nixl``. At 235B the network is far from saturated (~87 GB/s across 16 workers);
most of the sync is the rollout side loading each bucket into vLLM.

## Usage

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      backend: sharded_nixl
      update_weights_bucket_megabytes: 512
      engine_kwargs:
        sharded_nixl:
          device: cuda             # staging buffers: cuda (GPUDirect RDMA) or cpu (pinned host)
          rollout_dtype: bfloat16  # wire dtype of floating-point weights; match the rollout dtype
          ownership: adapter       # send each rollout worker only what it loads; "all" sends everything
```

It needs ``pip install nixl`` (tested with 1.4.1) and a trainer mode with a separate rollout, such as v1
``separate_async``.

## Limitations and roadmap

- Rank-local slices need vLLM with plain bf16, TP>1 and a local checkpoint; quantization, LoRA,
  speculative decoding and PP fall back to full tensors.
- The fail-closed marker lives in ``CheckpointEngineManager``, so a replica resumed through another
  handle (the ``fully_async`` rollouter's rebalance) bypasses it; moving it onto the rollout server actors
  would close that gap.
- Tested with FSDP and Megatron trainers, vLLM rollout and v1 ``separate_async``; SGLang, ``fully_async``
  and ``one_step_off`` are untested.
- On small dense models the per-sync overhead dominates and ``nccl`` stays faster; this is not investigated
  yet.
- verl CI does not install NIXL, so the GPU tests skip there; the planning code has CPU tests, and the vLLM
  bake test runs in the vLLM CI job.
