# MXFP4 MoE RL on Blackwell

Last updated: 10/29/2026

MXFP4 is the OCP microscaling FP4 format: E2M1 elements with one E8M0 scale per 32-element block. In verl it covers
the routed experts of MoE models. The rollout serves MXFP4 expert weights, quantized on the trainer side, and
DeepGEMM quantizes the expert activations to FP8 at run time. Training stays in BF16; with QAT, ModelOpt
fake-quantizes the routed experts during training: their weights exactly as the exported rollout weights, their
inputs to MXFP8, close to the rollout's FP8 activations.

The rollout quantization interface shared by all formats, the kernel defaults and the format details are described
in [Low-Precision Rollout and Quantization-Aware Training](qat.md). For MXFP8 end to end, see
[MXFP8 RL on Blackwell](mxfp8.md).

## Requirements

- Blackwell GPUs for the rollout: the MXFP4-weight × FP8-activation experts are served by vLLM's DeepGEMM FP4 MoE
  kernel (SM100/SM120). Hopper has no such kernel and is outside this recipe; see [qat.md](qat.md).
- Training: the Megatron-Bridge model path; QAT uses NVIDIA ModelOpt.
- Rollout: vLLM (the version verl locks).

## Configuration

```yaml
# MXFP4 rollout experts; dense layers stay in BF16
actor_rollout_ref.rollout.quantization:
  linear: null
  moe: MXFP4

# optional: MXFP4 QAT on the routed experts
actor_rollout_ref.actor.megatron.qat:
  enable: True
  mode: mxfp4        # mxfp4_mxfp8 also trains the dense layers with MXFP8 QAT (vLLM: linear: MXFP8)
```

Notes:

- `lm_head`, the token embedding, MoE routers and every dense layer stay in BF16 on the rollout and in training.
- During QAT, the routed experts' weights are fake-quantized with the same QDQ as the trainer-side exporter, so the
  QAT forward and the exported rollout weights agree bit for bit. The experts' inputs are fake-quantized to MXFP8
  (1x32 blocks), while the rollout's DeepGEMM kernel quantizes them to FP8 in 1x128 groups with E8M0 scales.
- QAT quantizes the Transformer Engine modules of the layer spec that Megatron-Bridge builds for the model, the
  grouped expert linears included, so tensor and expert parallelism can be combined.
- QAT and Transformer Engine FP8 training are mutually exclusive; the engine refuses the combination.
- QAT fake-quantizes all expert weights of a grouped linear in one call. On the example below, QAT makes a step about
  16% longer than the MXFP4 rollout alone; with ModelOpt's per-expert loop, the first QAT step took 3743 s instead
  of 1056 s on the same GPUs.

## Example: Qwen3-30B-A3B on DAPO-Math-17k

[`examples/mxfp4/run_qwen3_30b_a3b_megatron.sh`](https://github.com/verl-project/verl/blob/main/examples/mxfp4/run_qwen3_30b_a3b_megatron.sh)
is the MXFP4 counterpart of the MXFP8 Qwen3-30B-A3B example, with the same data and hyper-parameters on one
4-GPU Blackwell node (tested on 4×B300; 4×B200 untested): training TP2 × EP2 with full recompute and the whole
optimizer on the CPU, the rollout vLLM with TP2. `PRECISION` selects the arm: `mxfp4_qat` (MXFP4 rollout experts and
MXFP4 QAT), `mxfp4` (MXFP4 rollout experts only) and `bf16` (the baseline). The script reads DAPO-Math-17k in verl's
format from `TRAIN_FILE`, as
[verl-recipe's `dapo/prepare_dapo_data.sh`](https://github.com/verl-project/verl-recipe/blob/main/dapo/prepare_dapo_data.sh)
downloads it, and runs without a validation set.

```bash
PRECISION=mxfp4_qat TOTAL_TRAINING_STEPS=10 bash examples/mxfp4/run_qwen3_30b_a3b_megatron.sh
PRECISION=mxfp4 TOTAL_TRAINING_STEPS=10 bash examples/mxfp4/run_qwen3_30b_a3b_megatron.sh
PRECISION=bf16 TOTAL_TRAINING_STEPS=10 bash examples/mxfp4/run_qwen3_30b_a3b_megatron.sh
```

### Results

**Configuration**

- One 4×B300 machine for every arm, the same seed and data order, so steps pair across arms; 10 steps
- The script's hyper-parameters, as in the MXFP8 example: 64 prompts × 8 responses per step, prompts and responses up
  to 1024 and 8192 tokens, dynamic batching at 20,480 tokens per GPU, learning rate 1e-6; DAPO-Math-17k deduplicated
  to its 17,398 unique prompts
- The versions verl locks: vLLM 0.32.0, Transformer Engine 2.16.1, Megatron-Core 0.19.2, ModelOpt 0.44.0

"X/Y" is training precision X, rollout precision Y; BF16+QAT trains in BF16 with MXFP4 QAT on the routed experts.

| mean of steps 1–10 | BF16/BF16 | BF16/MXFP4 | BF16+QAT/MXFP4 |
|---|---|---|---|
| step time (s) | 829 | 890 | 1031 |
| generation (s) | 107 | 113 | 113 |
| `old_log_prob` (s) | 118 | 126 | 166 |
| `update_actor` (s) | 475 | 478 | 576 |
| `update_weights` (s) | 7 | 45 | 46 |
| `rollout_corr/kl` | 0.0017 | 0.0142 | 0.0031 |
| score | −0.191 | −0.157 | −0.156 |

Results and observations:

- QAT cuts the rollout–trainer mismatch of the MXFP4 rollout by 78%, to 1.8× the BF16 arm's. Per step,
  `rollout_corr/kl` ranged 0.0027–0.0036 with QAT, 0.0127–0.0164 without it and 0.0016–0.0019 in BF16.
- QAT makes a step 16% longer than the MXFP4 rollout alone: `old_log_prob` +40 s, `update_actor` +97 s.
- The MXFP4 rollout does not speed this job up: generation takes about as long as in BF16, and exporting the MXFP4
  expert weights adds about 38 s per step to `update_weights`.
- Ten steps are enough for speed and mismatch, not for the score.
