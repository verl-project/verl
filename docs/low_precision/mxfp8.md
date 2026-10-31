# MXFP8 RL on Blackwell

Last updated: 10/22/2026

MXFP8 is the OCP microscaling FP8 format: E4M3 elements with one E8M0 scale per 32-element block,
decoded natively by Blackwell tensor cores. This recipe trains with Transformer Engine's MXFP8 recipe and
serves the rollout in MXFP8, with the rollout's weights quantized on the trainer side with the learner's
block rule.

For FP8 on Hopper (the `blockwise` recipe), see [FP8 RL in verl](fp8.md). The rollout quantization
interface shared by all formats, and the vLLM and SGLang specifics, are described in
[Low-Precision Rollout and Quantization-Aware Training](qat.md).

## Requirements

- Blackwell GPUs (SM100+)
- Training: Megatron-Core >= 0.13 and Transformer Engine >= 2.1, on the Megatron-Bridge model path
- Rollout: vLLM 0.29 or SGLang 0.5.20 (the versions verl locks)

## Configuration

```yaml
# MXFP8 training via Transformer Engine
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "e4m3"                # element format; "hybrid" (e4m3 fwd + e5m2 bwd) also supported
  fp8_recipe: "mxfp8"        # 32-element block scaling

# MXFP8 rollout
actor_rollout_ref.rollout.quantization:
  linear: MXFP8
  moe: MXFP8
```

Notes:

- Weights stay in BF16 (`fp8_param` is not supported); only the GEMM inputs are cast to MXFP8, so
  checkpoints are unchanged.
- With `fp8_recipe: "mxfp8"`, verl pads packed sequences to the 32-token blocks MXFP8 needs.
- QAT and Transformer Engine FP8 training are mutually exclusive; the engine refuses the combination.
- `lm_head`, the token embedding and MoE routers stay in BF16 on the rollout, as in training. Further
  layers can be kept in BF16 with `rollout.quantization.ignore`.
- Given the same BF16 weight, the trainer-side exporter (compressed-tensors) and Transformer Engine pick
  the same E8M0 scale for every block except those whose amax is exactly `448 * 2**k`, where
  compressed-tensors picks the next one up; in such a block only elements below `amax / 14336`
  dequantize differently, by at most about `4.4e-6 * amax`. This compares the weight quantizers only, not
  activations, reduction order or the KV cache.

## Rollout kernels

verl defaults the MXFP8 dense and MoE kernels of vLLM and SGLang to DeepGEMM; for details, see
[Low-Precision Rollout and Quantization-Aware Training](qat.md).

## Checks

![The quantized-rollout sentinel, the post-sync self-check and the quantized-layer audit](https://github.com/user-attachments/assets/031f882a-9379-4ef7-bbce-3a746959d317)

| Check | Runs | Reports | Switch |
|---|---|---|---|
| Quantized-rollout sentinel | every training step | NaN log-probs, KL above 1.0, every response at the length cap | `VERL_QUANT_SENTINEL` |
| Post-sync self-check | after every weight sync | kernel scale layouts that did not follow the sync | `VERL_MXFP8_REFIT_CHECK` |
| Quantized-layer audit | training vs export: first weight sync after a training step; export vs engine: first weight sync, before each bucket is written | layers that training, the exported weights and the engine quantize differently | `VERL_QUANT_LAYER_AUDIT` |

Each switch takes `warn` (the default: log the finding and keep going), `raise` (stop the run or the sync)
or `0` (off). The sentinel applies to every quantized rollout, FP8 and FP4 included, and so does the audit's
export-vs-engine comparison (on SGLang, to the formats of its refit loader: MXFP8, MXFP4, NVFP4).

### Quantized-rollout sentinel

When the rollout is quantized, both trainers check each step's metrics:

- `training/rollout_probs_diff_mean` is non-finite while the batch had valid tokens;
- `rollout_corr/kl` is non-finite or above `VERL_QUANT_SENTINEL_KL_MAX` (default 1.0; healthy quantized
  rollouts read about 0.001–0.03);
- every response hits the length cap (`response_length/clip_ratio` close to 1) on
  `VERL_QUANT_SENTINEL_CLIP_STEPS` consecutive steps (default 2; `0` turns this rule off, and it is off
  under `rollout.ignore_eos`, where every response runs to the cap by design).

The rollout counts as quantized when `rollout.quantization` sets `linear`, `moe` or `config_file`, or the
model's own `config.json` (local paths only) carries a `quantization_config`. The sentinel catches gross
failures only: one layer type left unquantized (KL 0.007–0.010) stays within the healthy range.

### Post-sync self-check

After loading, both engines derive kernel-specific copies of the MXFP8 weights and scales (swizzled
scales, shuffled or packed experts). If a sync skips that step, the kernel pairs fresh weights with stale
scales while every shape and dtype still matches. After every sync, verl therefore runs the smallest MXFP8
linear layer and one expert of the smallest MXFP8 MoE layer through their own kernels on a small random
input and compares the output with a BF16 reference on the dequantized canonical weights. Healthy layers
read a relative error of about 0.03 (dense) and 0.05 (experts); a stale layout reads O(1).

- `VERL_MXFP8_REFIT_CHECK_TOL` (default 0.25) and `VERL_MXFP8_REFIT_CHECK_MOE_TOL` set the thresholds.
- On SGLang, the refit loader also reports experts whose scales the sync did not write.
- The expert probe is skipped, with a warning, where its made-up routing cannot send every row to one
  expert at weight 1. On vLLM: expert or data parallelism, non-SiLU gates, and, on monolithic MoE kernels, a
  routed scaling factor, grouped top-k, a score correction bias or custom routing. On SGLang: expert
  parallelism, TP without `reduce_results`, and non-softmax, grouped, biased or custom routing.

What it does not show:

- It samples one dense layer and one expert per sync; a layer outside the sample can still be stale.
- It runs the kernels eagerly. A re-derivation into new storage, instead of the storage a CUDA graph
  captured, passes the probe while graph replays keep reading the old values. That the captured storage is
  updated rests on indirect evidence (the re-derived layouts are copied back in place, and the KL stays in
  its healthy range), not on a graph replay.

### Quantized-layer audit

Three parties decide which layers are quantized, each on its own: training implicitly (TE linear modules
inside `fp8_autocast`), and by name the trainer-side exporter and the engine, both from the same quantization
config (`rollout.quantization`, or the checkpoint's own `quantization_config`). The audit compares them in two
steps, each once:

- **Training vs export**, when training runs a TE FP8 recipe, at the first weight sync after a training step
  (the sync before it has no FP8 trace yet): each decoder layer's FP8 trace (TE's FP8 weight workspaces)
  against the format the sync sends for that layer's GEMM projections (attention, MLP, experts,
  linear-attention and SSM in / out projections). Weights the exporter keeps in BF16 by design are skipped.
  Other decoder-layer weights, such as a convolution, are not judged, and the audit reports how many it left
  out.
- **Export vs engine**, whenever the rollout is quantized, during the first weight sync: inside the engine's
  weight loader, before each bucket is written, every incoming weight's format (quantized or high
  precision) against the engine parameter that receives it. vLLM checks every quantized format; SGLang
  checks the formats of its refit loader (MXFP8, MXFP4, NVFP4). Later syncs send the same formats and are
  not checked again.

`VERL_QUANT_LAYER_AUDIT` switches both. `VERL_QUANT_LAYER_AUDIT_TRACE=1` logs each step of the training-side
comparison. The training-side trace is per layer,
so a TE module kept in BF16 inside an FP8 layer would go unnoticed (Megatron has no such per-module option
today). With `disable_parameter_transpose_cache=True` TE keeps no FP8 workspace, and the audit warns once
that it cannot run.

### Configurations SGLang refuses

Two configurations are refused before a weight sync writes anything:

- MoE layers kept in BF16 (`rollout.quantization.ignore`) whose experts the MoE runner re-laid out at load:
  SGLang's FlashInfer TRT-LLM runner stores BF16 experts in a block layout a weight sync cannot write, while
  verl's default DeepGEMM runner keeps them in checkpoint layout.
- The delta checkpoint engine with any quantized rollout (`rollout.quantization` or the checkpoint's own
  `quantization_config`): it applies its payloads without the trainer-side weight quantization.

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for an SGLang rollout: verl starts SGLang
with `enable_memory_saver=True`, which refuses that allocator mode.

## Example: Qwen3-8B on GSM8K

[`examples/mxfp8/run_qwen3_8b_megatron.sh`](https://github.com/verl-project/verl/blob/main/examples/mxfp8/run_qwen3_8b_megatron.sh)
runs GRPO on Qwen3-8B with GSM8K, MXFP8 end to end; `PRECISION=bf16` runs the same job in BF16. Its defaults
fit one B200: the whole optimizer stays on the CPU, and parameters are offloaded while the rollout runs.

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
PRECISION=mxfp8 bash examples/mxfp8/run_qwen3_8b_megatron.sh
PRECISION=bf16 bash examples/mxfp8/run_qwen3_8b_megatron.sh
```

### Results

**Configuration**

- One B200; verl's locked environment (vLLM 0.29.0, Transformer Engine 2.16.1, Megatron-Core 0.19.2,
  Megatron-Bridge 0.6.2)
- The script's defaults: 64 prompts × 8 responses per step, prompts and responses up to 1024 and 2048 tokens,
  learning rate 1e-6; 10 steps (`TOTAL_TRAINING_STEPS=10`)
- Validation on the GSM8K test set (1,319 problems, greedy decoding) at steps 0, 5 and 10
- vLLM rollout with FlashInfer available to the workers (`nvcc` on their `PATH`), where vLLM picks FlashInfer
  CuTe-DSL for the MXFP8 dense GEMMs

![Qwen3-8B on GSM8K, BF16 vs MXFP8](https://github.com/user-attachments/assets/58bc248c-91cf-4f3f-bff5-64f3052230d1)
*Blue: BF16, vermillion: MXFP8 end to end. wandb: [BF16](https://forge.coreweave.com/wandb/wengeezhang-bytedance/verl_grpo_gsm8k_mxfp8/runs/zodbg1pr), [MXFP8](https://forge.coreweave.com/wandb/wengeezhang-bytedance/verl_grpo_gsm8k_mxfp8/runs/cr5dk8fs)*

Results and observations:

- GSM8K test accuracy at steps 0, 5 and 10 is 0.372, 0.616 and 0.803 with MXFP8, against 0.329, 0.610 and
  0.819 with BF16, and the training reward follows the BF16 curve. Both are single runs of 10 steps, too few
  to resolve the 1.6-point gap at step 10 (21 of 1,319 problems).
- The mismatch between the rollout and the trainer (`rollout_corr/kl`) is 0.0063–0.0071 with MXFP8, against
  0.0006–0.0008 with BF16, and stays flat over the run.

## Example: Qwen3-30B-A3B on DAPO-Math-17k

[`examples/mxfp8/run_qwen3_30b_a3b_megatron.sh`](https://github.com/verl-project/verl/blob/main/examples/mxfp8/run_qwen3_30b_a3b_megatron.sh)
runs GRPO on Qwen3-30B-A3B (MoE) with DAPO-Math-17k on one 4×B200 node, MXFP8 end to end, experts included;
`PRECISION=bf16` runs the same job in BF16. Training runs TP2 × EP2 with full recompute and the whole optimizer on the
CPU, the rollout vLLM with TP2. The script reads DAPO-Math-17k in verl's format from `TRAIN_FILE`, as
[verl-recipe's `dapo/prepare_dapo_data.sh`](https://github.com/verl-project/verl-recipe/blob/main/dapo/prepare_dapo_data.sh)
downloads it, and runs without a validation set.

```bash
PRECISION=mxfp8 TOTAL_TRAINING_STEPS=10 bash examples/mxfp8/run_qwen3_30b_a3b_megatron.sh
PRECISION=bf16 TOTAL_TRAINING_STEPS=10 bash examples/mxfp8/run_qwen3_30b_a3b_megatron.sh
```

### Results

**Configuration**

- One 4×B200 machine for every arm, the same seed and data order, so steps pair across arms; 10 steps
- The script's hyper-parameters: 64 prompts × 8 responses per step, prompts and responses up to 1024 and 8192
  tokens, dynamic batching at 20,480 tokens per GPU, learning rate 1e-6; DAPO-Math-17k deduplicated to its 17,398
  unique prompts; FlashInfer autotuning off
- An earlier version of this integration: vLLM 0.24 served MXFP8 through ModelOpt and FP8 blockwise through its own
  FP8 method, before the rollout weights were quantized on the trainer side with compressed-tensors

"X/Y" is training precision X, rollout precision Y: FP8-TW is Transformer Engine's per-tensor FP8 with a BF16
rollout, FP8-BW verl's FP8 blockwise recipe end to end.

| mean of steps 1–10 | BF16/BF16 | FP8-TW/BF16 | FP8-BW/FP8-BW | MXFP8/MXFP8 |
|---|---|---|---|---|
| step time (s) | 506 | 487 | 489 | 452 |
| generation (s) | 134 | 133 | 115 | 102 |
| `update_actor` (s) | 244 | 225 | 241 | 221 |
| `rollout_corr/kl` | 0.0018 | 0.0058 | 0.0090 | 0.0045 |
| score | −0.130 | −0.150 | −0.169 | −0.160 |

![Qwen3-30B-A3B on DAPO-Math-17k, per step](https://github.com/user-attachments/assets/569f6f0f-1b78-4e62-9401-6c02c5cad49e)
*Per step on one 4×B200 machine. wandb: [verl_mxfp8_b200](https://forge.coreweave.com/wandb/wengeezhang-bytedance/verl_mxfp8_b200), group "D: Qwen3-30B-A3B, DAPO-Math-17k"*

Results and observations:

- MXFP8 cuts the step time by 11% against BF16 (generation −24%, `update_actor` −9%), more than either FP8 recipe,
  with the lowest rollout–trainer mismatch of the three low-precision arms.
- Ten steps are enough for speed and mismatch, not for the score.
- The MXFP8 arm, rerun for 5 steps on the compressed-tensors path (vLLM 0.29, verl's locked environment, another
  4×B200 machine), read `rollout_corr/kl` 0.0037–0.0040.

## Qwen3-8B on DAPO-Math-17k

The dense model on the same data and per-step hyper-parameters: 2×B200 (TP2), dynamic batching at 12,288 tokens per
GPU, 32 steps, the earlier version of the integration as above. Two more differences:

- Every arm, BF16 included, ran vLLM with an FP8 KV cache (`kv_cache_dtype=fp8`; vLLM 0.24 then also quantizes Q).
  It raises the mismatch of every arm, so these KL values do not compare with the other runs on this page.
- The MXFP8 rollout kept `gate_up_proj` in BF16, through a layer-exclusion bug fixed since.

| mean of steps 23–32 | BF16/BF16 | FP8-TW/BF16 | MXFP8/MXFP8 |
|---|---|---|---|
| `rollout_corr/kl` | 0.0088 | 0.0297 | 0.0088 |
| entropy | 0.220 | 0.083 | 0.338 |
| score | 0.150 | 0.240 | 0.135 |
| response length | 6198 | 5748 | 6202 |
| step time (s) | 407 | 345 | 386 |

![Qwen3-8B on DAPO-Math-17k, per step](https://github.com/user-attachments/assets/444018cd-2219-49fe-a907-8b147ef9b804)
*Per step on 2×B200, with an FP8 KV cache in every arm. wandb: [verl_mxfp8_b200](https://forge.coreweave.com/wandb/wengeezhang-bytedance/verl_mxfp8_b200), group "B: Qwen3-8B, DAPO-Math-17k"*

Results and observations:

- The MXFP8 arm's mismatch stays at the BF16 arm's level, while FP8-TW with a BF16 rollout drifts from 0.009 at
  step 1 to 0.037 at step 32.
- Entropy falls by a third with BF16 (0.304 at step 1, 0.206 at step 32), nearly collapses with FP8-TW (0.307,
  0.062), and rises slightly with MXFP8 (0.309, 0.344). FP8-TW's higher score and shorter responses come with that
  collapse; without a validation set, these runs do not show which arm generalizes better.
