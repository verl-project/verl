Adding DeepSeek V4 support
==========================

Last updated: 10/26/2026.

This guide describes the DeepSeek-V4-Flash integration for a Megatron actor and
a vLLM rollout model. Most of the model-specific work is at two boundaries:

* synchronizing quantized weights from Megatron to vLLM; and
* replaying expert routes between rollout, log-probability computation, and
  actor updates.

Weight synchronization is required for online training. Router replay is a
separate feature and is required only when R2 or R3 is enabled. DeepSeek V4
also needs a few independent compatibility fixes for model construction and
fused kernels.

Synchronizing quantized weights
-------------------------------

DeepSeek-V4-Flash does not use a single FP8 representation for all weights:

* Dense weights contain E4M3 FP8 values with UE8M0 scales.
* Routed MoE experts contain packed FP4 weights.
* During initial loading, vLLM converts the raw expert ``w13``, ``w2``, and
  scale tensors into an MXFP4 or MegaMoE layout. This conversion can replace
  or remove the original parameters.

Megatron exports subsequent updates in the raw checkpoint layout. Those
tensors therefore cannot be copied directly into the post-processed vLLM
parameters, even if their names match.

The update path restores the representation expected by vLLM's
``load_weights`` method before accepting an update:

1. Restore the raw dense and expert parameters together with their
   ``weight_loader`` metadata.
2. Load every weight bucket before running any non-idempotent model
   post-processing.
3. Preserve UE8M0 scales in their original dtype instead of dequantizing and
   requantizing them through another floating-point format.
4. Let the vLLM loaders apply tensor-parallel sharding and merge ``w1`` and
   ``w3`` into ``w13``.
5. After the final bucket, rebuild the MXFP4 or MegaMoE representation once.

Running post-processing after each bucket is incorrect because later buckets
would be loaded into parameters that an earlier post-processing pass already
transformed. The bucketed transfer path must prepare the model before the first
bucket and finalize it only after the last bucket.

``verl/utils/vllm/vllm_quant_utils.py`` is the entry point: it quantizes the
outgoing weight stream and dispatches the prepare/finalize pair to the scheme
that owns each layer. ``verl/utils/vllm/vllm_fp8_utils.py`` handles the block
FP8 linear and MoE methods, ``verl/utils/vllm/vllm_fp4_utils.py`` handles the
MXFP4 routed experts, and the bucket-level orchestration is in
``verl/workers/rollout/vllm_rollout/utils.py``.

Incorrect synchronization can fail immediately with an expert shape mismatch
such as ``target 1024 vs loaded 2048``. It can also complete ``load_weights``
while leaving the expert layout or scales incorrect. In that case, the visible
symptom is a severe regression in rollout/actor log-probability correlation.
A successful load is therefore not sufficient evidence that synchronization
is correct.

Replaying expert routes
-----------------------

R2 records routes during actor log-probability computation and replays them
during the actor update. R3 records routes in the rollout backend and replays
them in Megatron. Neither mode is needed merely to run DeepSeek V4; they are
alignment features enabled by configuration.

The first three routed layers in DeepSeek-V4-Flash use a hash router. Their
expert IDs come from ``input_ids`` and a token-to-expert table rather than from
learned router logits. As a result, the existing learned top-k interception
does not observe these layers.

Supporting replay requires the Megatron path to:

1. Pass ``input_ids`` into the decoder so the hash router has the same input as
   the rollout model.
2. Record the three hash-router outputs in addition to the later learned
   top-k-router outputs.
3. Preserve routed-layer order across vLLM and Megatron.

Without the hash-router entries, vLLM reports routes for every MoE layer while
Megatron omits the first three. Every subsequent route is then replayed against
the wrong layer, even when the route tensors have compatible shapes.

R3 also needs a causal replay mask. The mask must include rows that influence
response-token logits, not only rows marked as response tokens. When a replayed
top-k result contains duplicate expert indices, dispatcher token counts must be
derived from the resulting routing map rather than assumed to equal
``num_tokens * topk``.

The decoder input and hash-router interception are implemented in
``verl/models/mcore/model_forward_fused.py`` and
``verl/utils/megatron/router_replay_patch.py``. R3 mask construction and route
distribution are implemented in
``verl/utils/megatron/router_replay_utils.py``.

Model and kernel compatibility
------------------------------

The following requirements are independent of weight synchronization and
router replay:

* Some Transformers releases do not register the ``deepseek_v4`` model type.
  Verl uses the vLLM configuration only for that exact missing-model-type case;
  unrelated configuration errors continue to propagate.
* When MTP is disabled in Verl, it must also be disabled in the Megatron Bridge
  provider. The associated CSA metadata must be trimmed to match the resulting
  layer count.
* The fused DSA kernel requires each local THD shard to contain at least one CSA
  window. Shorter local shards must be padded before the kernel call and
  unpadded afterward.
* The DeepSeek V4 decoder may return a tuple. The fused forward path accepts
  that tuple without changing the tensor-only behavior used by existing
  models.

Verification
------------

Verify the integration in this order:

1. Disable router replay and run at least two optimizer steps. The second
   actor-to-rollout update exercises synchronization after vLLM has already
   post-processed its parameters once.
2. Compare rollout and actor token log probabilities. Report Pearson
   correlation together with maximum, mean, and standard-deviation
   differences. A large mismatch at this stage indicates a weight, scale, or
   packed-layout problem rather than a replay problem.
3. Enable the intended R2 or R3 mode and repeat the comparison. Verify that the
   three hash-routed layers and all learned-router layers are recorded and
   replayed in the same order.
4. Save and reload a training checkpoint containing model, optimizer,
   scheduler, and trainer state. Successful weight transfer does not validate
   optimizer checkpointing for mixed FP32 and quantized parameters.
5. Run with prompt and response lengths representative of the target workload.
   Short smoke tests can miss sequence-packing, attention-window, and routed
   expert failures.

Running DeepSeek-V4-Pro at scale
--------------------------------

DeepSeek-V4-Pro (61 layers, 384 routed experts, ~1.6 T parameters) runs through
the same integration as Flash; what changes is scale. The reference launcher is
``examples/grpo_trainer/run_deepseek_v4_pro_megatron.sh`` with two hardware
profiles, ``HW=h100`` (64 nodes x 8 H100 80 GB) and ``HW=gb200`` (64 nodes x
4 GB200). Both were validated with GRPO on DAPO-Math-17k; the numbers below are
single-variable follow-ups on top of the respective profile.

**Topology.** DSV4 fixes ``tensor_model_parallel_size=1`` on the Megatron side,
so a Pro actor is sharded with pipeline and expert parallelism only: PP32 x EP8
(DP2) on 512 H100s, PP16 x EP16 (DP1) on 256 GB200s. Stage 0 of the pipeline
must hold the embedding and the three hash-routed layers; Megatron-Bridge
generates the layout for PP16, the PP32 layout is spelled out in the launcher.
A vLLM replica of Pro does not fit in one node, so the rollout uses cross-node
data-parallel replicas (TP8 x DP8 x EP64 on H100, TP1 x DP32 x EP32 on GB200)
under verl's ``expert_parallel_size = TP x DP`` rule. This needs verl ``main`` at or
after #8234 (merged 2026-10-26): before it, vLLM 0.29 rejected the headless engines
verl starts on the other nodes of a replica.

**Memory.** With ``param_offload`` and ``optimizer_offload`` the GPU is rarely
the constraint: the actor peaks at ~26 GB per H100 (response 4096, EP8; 77 GB
per GPU including the sleeping vLLM worker on the stage-0 nodes) and 34-63 GB
per GB200. The exception is response length 8192 on 80 GB GPUs, where
stage 0 (embedding plus the three hash layers, ~11.6 B parameters at EP8) does
not fit next to the ~17 GiB vLLM keeps resident after sleep; ``ACTOR_EP=16``
(DP1) halves it. The host is the usual limit, because Megatron's
HybridDeviceOptimizer keeps fp32 masters and Adam moments for every rank on the
CPU (~12 bytes per parameter per rank, independent of DP). Three levers were
needed:

* ``override_ddp_config.grad_reduce_in_fp32=False`` (bf16 grad buffers); on
  80 GB GPUs stage 0 does not fit without it.
* ``use_precision_aware_optimizer=True`` with ``main_grads_dtype=bf16``: fp32
  master shards move from the GPU into the offloaded optimizer state. Required
  on H100, optional on GB200 (-30 % GPU and -22 % host memory at 4 layers,
  ~+8 % step time). DSV4 then produces mixed bf16/fp32 gradient groups, which
  Megatron-LM's grad-norm kernels do not accept until
  https://github.com/NVIDIA/Megatron-LM/pull/7279 lands, and the quantized
  main-param branch builds non-leaf shards until
  https://github.com/NVIDIA/Megatron-LM/pull/8245 lands. The Megatron engine
  applies runtime equivalents of both (``verl/models/mcore/patch.py``;
  ``VERL_MEGATRON_OPTIMIZER_COMPAT=0`` disables them).
* ``optimizer_offload_fraction < 1`` keeps part of the Adam state on the GPU
  (fixed by https://github.com/verl-project/verl/pull/8286).

Keep the Ray object store small (32 GiB) when starting the cluster; its default
of 30 % of host memory competes with the optimizer.

**Checkpoints.** ``save_contents=[model,extra]`` is the default of the launcher:
the optimizer state of Pro is ~22 TB, and resuming an offloaded optimizer needs
verl ``main`` at or after #8235 (merged 2026-10-26). A model-only save is an HF
export through Megatron-Bridge (~800 GB, ~30 min on 64 H100 nodes) or, when the
actor was loaded with ``use_dist_checkpointing``, a bf16 Megatron
dist-checkpoint (2.9 TB, ~40 s). On GB200 the save needs the precision-aware
optimizer or 128 nodes (PP16 x EP32); otherwise the head node runs out of host
memory during the save.

**Rollout kernels.** verl defaults low-precision rollouts to DeepGEMM
(``moe_backend=deep_gemm``); the Pro launcher overrides the MoE kernel to Marlin
(W4A16) on both profiles. vLLM 0.29's DeepGEMM MXFP4 kernel is SM100+ only, so on
H100 every vLLM worker fails at ``load_model`` with the default, and on GB200 the
same kernel died with an illegal memory access in vLLM's profiling run at 10,240
batched tokens (prompt 2048 + response 8192); where it ran (3,072 batched tokens)
it was 5-9 % slower per step than Marlin for 64 samples. The other MXFP4 kernels
are not options for a native DSV4 checkpoint: FlashInfer TRT-LLM cannot be
refitted, and the FlashInfer CUTLASS kernel is not selectable (the CLI accepts
only its BF16-activation variant, which rejects the MXFP4 weight scheme). The
dense FP8 layers keep verl's DeepGEMM default (``linear_backend=deep_gemm``).

**What the reward means.** On DAPO-Math-17k the per-step
``critic/score/mean`` of Pro is almost exactly ``1 - 2 x clip_ratio``: in 192
inspected samples every response that reached a final answer was correct and
every truncated one was scored -1. The score therefore tracks the truncation
rate, not accuracy; track the accuracy of completed responses separately
(``trainer.rollout_data_dir``) and expect to need harder prompts or a larger
response budget before the curve moves.

**Long runs.** On clusters with a wall-clock limit, run the job as a chain of
segments with ``trainer.save_freq`` and ``trainer.resume_mode=auto``; each
segment resumes from the newest ``global_step_*`` and the first step after a
resume costs 1.5-4x a steady step (initial weight sync plus every JIT the new
job has to redo; ~2x on GB200, ~3.7x on H100).

Measured on the two profiles (verl ``main`` a4b9354 + #8234, Megatron-Bridge 0.6.2, vLLM 0.29.0, Marlin MoE kernels, R3, 3 GRPO steps each):

.. list-table::
   :header-rows: 1

   * - Profile
     - Response length
     - Steady step time
     - of which ``gen`` / ``update_weights``
     - rollout/actor log-prob Pearson (R3)
     - Host peak per node
   * - ``h100`` (64 x 8 H100, PP32 x EP8, rollout TP8 x DP8)
     - 4096
     - ~690 s (first step 2585 s with JIT)
     - ~280 s / ~235 s
     - 0.998
     - 558 GB on the two pipeline-stage-0 nodes, 244-409 GB on the others (mean 383 GB)
   * - ``gb200`` (64 x 4 GB200, PP16 x EP16, rollout DP32)
     - 8192
     - ~965 s (first step 1991 s)
     - ~500 s / ~251 s
     - 0.998
     - 929 GB on the head node (stock optimizer; no headroom for a save at this shape, see Checkpoints)
