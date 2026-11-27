# MXFP8 RL on AMD ROCm

Last updated: 11/27/2026

AMD Instinct MI350 / MI355 (gfx950) decode the OCP MXFP8 format natively: E4M3 elements with one E8M0 scale per
32-element block. This page records running the [MXFP8 RL](mxfp8.md) recipe, Transformer Engine's MXFP8 training
recipe with an MXFP8 rollout, on gfx950, model by model. The rollout quantization interface is described in
[Low-Precision Rollout and Quantization-Aware Training](qat.md).

## Requirements

- AMD Instinct MI350 / MI355 (gfx950)
- The `docker/rocm/Dockerfile.rocm103` image (ROCm 10.3, built on `rocm/primus:v26.10` with vLLM 0.34), with the vLLM
  patches in `docker/rocm/patches/` applied as the Dockerfile describes
- `NVTE_ROCM_ENABLE_MXFP8=1`

## Configuration

The same as on Blackwell (see [MXFP8 RL](mxfp8.md#configuration)), plus the ROCm Megatron attention backend:

```yaml
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "e4m3"
  fp8_recipe: "mxfp8"
  attention_backend: fused   # the image exports NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=1

actor_rollout_ref.rollout.quantization:
  linear: MXFP8
  moe: MXFP8
```

Notes:

- On gfx950 Transformer Engine runs an MXFP8 GEMM on HipKittens only when its token and feature dimensions are
  multiples of 256 (K a multiple of 128); otherwise rocRoller generates a kernel on the CPU at run time, minutes per
  kernel. verl pads each expert's tokens to 256 for MXFP8 on ROCm (`moe_router_padding_for_fp8=True`); the example
  script sets the remaining options (below).
- The reference model runs in MXFP8 too. The example script sets its MXFP8 options explicitly: setting any key of
  the ref's `override_transformer_config` replaces the dict it would otherwise inherit from the actor.
- `VLLM_ROCM_MXFP8_BLOCK32_SCALES=0` (the image's default; the example script exports it too) keeps vLLM's per-row
  MXFP8 weight scales, so the weight refit works after vLLM starts on dummy weights.

## Rollout kernels

The rollout runs vLLM with AITER enabled: `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1`.

## Example: Qwen3-30B-A3B on DAPO-Math-17k

`examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh` is the ROCm version of the
[Blackwell example](mxfp8.md#example-qwen3-30b-a3b-on-dapo-math-17k) for one 4×MI355X node: GRPO on Qwen3-30B-A3B
(MoE, 128 experts, top-8) with DAPO-Math-17k, MXFP8 end to end, experts included; `PRECISION=bf16` runs the same job
in BF16. The task settings are the upstream ones: 64 prompts × 8 responses per step, prompts and responses up to 1024
and 8192 tokens, dynamic batching at 20,480 tokens per GPU, learning rate 1e-6, KL loss 0.001 (`low_var_kl`), no
validation set. The script reads DAPO-Math-17k in verl's format from `TRAIN_FILE`.

```bash
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1   # rollout MoE on AITER, as in the results below
COMMON=(SEED=42 TOTAL_TRAINING_STEPS=50 MODEL_PATH=Qwen/Qwen3-30B-A3B TRAIN_FILE=$HOME/data/dapo-math-17k/train.parquet)
env "${COMMON[@]}" PRECISION=mxfp8 bash examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh
env "${COMMON[@]}" PRECISION=bf16 bash examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh
```

`SEED` fixes the data order so steps pair across runs. The script's defaults fit one 4×MI355X node: training
TP1 × EP4 × ETP1 with selective recompute of `core_attn` and `moe_act` and the whole optimizer on the CPU; the rollout
on vLLM with TP2 and `gpu_memory_utilization=0.4`. By precision it sets:

| `PRECISION` | Settings |
|---|---|
| `mxfp8` | `NVTE_ROCM_ENABLE_MXFP8=1`; experts on HipKittens' grouped MXFP8 GEMM (`NVTE_USE_HIPKITTENS_GROUPED_GEMM=1`); `moe_router_padding_for_fp8=True`; `pad_to_length=True` with `PAD_TO_LENGTH_BUCKET=512` for the dense GEMMs; the reference model in MXFP8 too |
| `bf16` | experts on Transformer Engine's CK grouped GEMM (`NVTE_USE_CK_GROUPED_GEMM=1`) |

### Results

**Configuration**

- One 4×MI355X machine, the same four GPUs for both runs, back to back; `SEED=42`, so steps pair across runs;
  50 steps
- The script's defaults (above); the rollout MoE on AITER (`VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1`)
- The `docker/rocm/Dockerfile.rocm103` image with its vLLM 0.34 patches
- DAPO-Math-17k deduplicated to its 17,398 unique prompts
- W&B project [wei-cai/verl_mxfp8_rocm_mi355](https://wandb.ai/wei-cai/verl_mxfp8_rocm_mi355); runs
  [BF16 (`qwen3_30b_a3b_dapo_bf16_mi355x4`)](https://wandb.ai/wei-cai/verl_mxfp8_rocm_mi355/runs/q4ts3xu6) and
  [MXFP8 (`qwen3_30b_a3b_dapo_mxfp8_mi355x4`)](https://wandb.ai/wei-cai/verl_mxfp8_rocm_mi355/runs/y11llnji)

"X/Y" is training precision X, rollout precision Y. Step 1 is left out of the means: it includes one-time loading and
compilation (412 s for BF16, 483 s for MXFP8).

| mean of steps 2–50 | BF16/BF16 | MXFP8/MXFP8 |
|---|---|---|
| step time (s) | 339.3 | 334.2 |
| generation (s) | 150.6 | 138.2 |
| `old_log_prob` (s) | 38.1 | 39.4 |
| ref (s) | 36.9 | 37.6 |
| `update_actor` (s) | 111.1 | 113.2 |
| `update_weights` (s) | 2.3 | 5.5 |
| throughput (tokens/s/GPU) | 2314 | 2327 |
| `rollout_corr/kl` | 0.0029 | 0.0077 |
| score | 0.122 | 0.133 |
| response length (tokens) | 5984 | 5917 |
| `max_memory_reserved` (GB) | 189 | 200 |

| | BF16/BF16 | MXFP8/MXFP8 |
|---|---|---|
| score, mean of steps 2–10 → 41–50 | −0.071 → 0.289 | −0.084 → 0.304 |
| response length, steps 2–10 → 41–50 | 6226 → 5493 | 6218 → 5394 |
| `rollout_corr/kl`, steps 2–10 → 41–50 | 0.0018 → 0.0038 | 0.0048 → 0.0097 |

Results and observations:

- MXFP8 is slightly faster than BF16 end to end on 4×MI355X: the step takes 1.5% less (the per-step ratio has a
  median of 0.979, and MXFP8 is faster in 43 of the 49 paired steps) and throughput is 0.5% higher. Generation is
  8.2% shorter; `update_actor` takes 1.9% longer, `old_log_prob` and ref 2–3% longer, and `update_weights` 3.2 s
  longer.
- Training follows BF16 over the 50 steps: the score rises from about −0.08 to 0.29–0.30 and the responses shorten
  from about 6,200 to 5,400–5,500 tokens in both runs.
- The rollout–trainer mismatch grows in both runs as training proceeds and stays at 2.5–2.7× BF16's (0.0048 against
  0.0018 over steps 2–10, 0.0097 against 0.0038 over steps 41–50).
