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
- Validated with PyTorch 2.13 / CUDA 13, Transformer Engine 2.20.2, Megatron-Core 0.19.2, Megatron-Bridge 0.6.2
  and vLLM 0.32.0 (`uv sync --frozen --extra megatron --extra vllm`)
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

- `fp4` applies only to the routed experts and cannot be combined with `qat`. Training quantizes their
  activations per token in the forward pass and runs the backward GEMMs on the dequantized forward operands;
  `NVTE_BACKWARD_OVERRIDE=high_precision` uses the BF16 operands instead.
- The rollout must keep exactly the training BF16 layers; the trainer checks every exported expert weight
  against them.
- Transformer Engine reads its NVFP4 recipe from `NVTE_*` environment variables; the trainer rejects a
  recipe that differs. See `examples/real_nvfp4/runtime_env.yaml`.
- Routed experts are found by name, so the recipe is not tied to one architecture. It is validated on
  Qwen3-30B-A3B, Qwen3.5-35B-A3B and a small model with the DeepSeek-V3 layout (dense first layer, MLA,
  shared expert).

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

## Example: Qwen3-30B-A3B-Base with DAPO

[`examples/real_nvfp4/run_qwen3_30b_megatron.sh`](https://github.com/verl-project/verl/blob/main/examples/real_nvfp4/run_qwen3_30b_megatron.sh)
runs DAPO with routing replay (R3) and token-level TIS on 8 nodes × 4 GB200, keeping the first two and last four
of the 48 layers in BF16; `PRECISION_MODE=bf16` runs the same job in BF16. With a Ray cluster running:

```bash
MODEL_PATH=/shared/models/Qwen3-30B-A3B-Base \
TRAIN_FILE=/shared/data/train.parquet \
TEST_FILE=/shared/data/validation.parquet \
PRECISION_MODE=real_nvfp4 EXP_NAME=qwen3-nvfp4 NNODES=8 \
  bash examples/real_nvfp4/run_qwen3_30b_megatron.sh
```

See the [example README](https://github.com/verl-project/verl/blob/main/examples/real_nvfp4/README.md) for the
dataset format, runtime environment and tests.

### Results

700 steps of NVFP4 and BF16 with the example script, on an earlier version of this integration (vLLM 0.29).

![Training score](https://github.com/user-attachments/assets/d0693c0b-6789-4285-8658-44a885d72a72)
*`critic/score/mean`. Orange: NVFP4, red: BF16*

![Entropy](https://github.com/user-attachments/assets/e63b552c-fe9b-4b23-9751-62c38976732c)
*`actor/entropy`. Orange: NVFP4, red: BF16*

![Rollout-training mismatch](https://github.com/user-attachments/assets/13131fe5-c670-41ea-91e1-68e5fff123ee)
*`rollout_corr/kl`. Orange: NVFP4, red: BF16*

![Gradient norm](https://github.com/user-attachments/assets/b121c562-b6f7-4432-8a7e-ed5768d78e69)
*`actor/grad_norm`. Orange: NVFP4, red: BF16*

Results and observations:

- The training score follows BF16 for the first ~450 steps and is lower afterwards.
- The rollout–training mismatch is 0.003–0.01 with NVFP4, against about 0.0003 with BF16.
- NVFP4 rollout generation is about 1.3× faster than BF16 once the mean response exceeds ~2.5k tokens, giving
  roughly 1.2× shorter steps; NVFP4 training itself is still slower than BF16 in Transformer Engine.
