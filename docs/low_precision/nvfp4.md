# NVFP4 RL on Blackwell (experimental)

Last updated: 10/28/2026

This recipe trains the routed experts of an MoE model in real NVFP4 with Megatron and Transformer Engine,
and serves the rollout in the same precision. Activations are quantized per token on both sides.
Parameters stay BF16; attention, dense MLPs, shared experts, router, norms, embeddings and the LM head run
in BF16.

For NVFP4 fake-quantization training, see [NVFP4 QAT](nvfp4_qat.md). The rollout quantization interface
shared by all formats is described in [Low-Precision Rollout and Quantization-Aware Training](qat.md).

## Requirements

- Blackwell (SM100) GPUs
- Megatron backend, PP=1 without virtual pipeline parallelism
- vLLM rollout with TP and EP=1 or EP=TP, the colocated `naive` checkpoint engine and `FULL_DECODE_ONLY`
  CUDA graphs. Data-parallel EP, EPLB, context parallelism (DCP/PCP) and speculative decoding are not
  supported.

## Configuration

```yaml
actor_rollout_ref:
  actor:
    megatron:
      override_transformer_config:
        fp4: e2m1
        fp4_recipe: nvfp4
        first_last_layers_bf16: true
        num_layers_at_start_in_bf16: 2   # whole decoder layers kept in BF16
        num_layers_at_end_in_bf16: 4
  rollout:
    quantization:
      moe: NVFP4_PER_TOKEN
      ignore: ["re:.*layers\\.(0|1|44|45|46|47)\\.mlp\\.experts\\..*"]  # same layers, 48-layer model
```

- `fp4` applies only to the routed experts and cannot be combined with `qat`.
- The rollout must keep exactly the training BF16 layers; the trainer checks every exported expert weight
  against them.
- Transformer Engine reads its NVFP4 recipe from `NVTE_*` environment variables; the trainer rejects a
  recipe that differs. See `examples/real_nvfp4/runtime_env.yaml`.

## Rollout

The `NVFP4_PER_TOKEN` preset exports NVFP4 expert weights on the trainer side, bit-identical to vLLM's
online `nvfp4_per_token` quantization, and runs them on FlashInfer's TRT-LLM MoE kernel with per-token
activation scales. Refits load each bucket in place into the storage the CUDA graphs captured, so the extra
memory is one MoE layer's post-processing. With EP off, each rank computes the FC2 per-token scale over its
shard of the expert.

## Mismatch

Qwen3.5-35B-A3B, EP=4 on 4 GPUs, vLLM rollout TP=1, measured with the script in
[qat.md](qat.md#measuring-the-mismatch):

| Rollout | Training | Backend | k3_kl |
|---|---|---|---|
| NVFP4 MoE | NVFP4 QAT | FlashInfer CUTLASS (MoE; BF16 Linear) | 0.00832 |
| NVFP4_PER_TOKEN MoE | BF16 | FlashInfer TRT-LLM (MoE; BF16 Linear) | 0.01025 |
| NVFP4_PER_TOKEN MoE | native NVFP4 | FlashInfer TRT-LLM (MoE; BF16 Linear) | 0.00740 |

## Example

`examples/real_nvfp4` trains Qwen3-30B-A3B-Base with DAPO on 8 nodes and lists the runtime versions and
tests.
