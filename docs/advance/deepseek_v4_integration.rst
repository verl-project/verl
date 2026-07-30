Adding DeepSeek V4 support
==========================

Last updated: 07/12/2026.

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

The DeepSeek V4 parameter restoration, loaders, and scale handling are in
``verl/utils/vllm/vllm_dsv4_fp8_utils.py``. The generic FP8 dispatch is in
``verl/utils/vllm/vllm_fp8_utils.py``, and the bucket-level orchestration is in
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
