# MXFP4 MoE RL on Blackwell

Last updated: 10/28/2026

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
- QAT currently costs much more time than the BF16 arm: on Qwen3-30B-A3B with this example (4×B300, TP2 × EP2), the
  first step's QAT forward (`old_log_prob`) took about 10× the reference forward of the same model (`ref`).

## Example: Qwen3-30B-A3B on DAPO-Math-17k

[`examples/mxfp4/run_qwen3_30b_a3b_megatron.sh`](https://github.com/verl-project/verl/blob/main/examples/mxfp4/run_qwen3_30b_a3b_megatron.sh)
is the MXFP4 counterpart of the MXFP8 Qwen3-30B-A3B example, with the same data and hyper-parameters on one
4-GPU Blackwell node (tested on 4×B300; 4×B200 untested): training TP2 × EP2 with full recompute and the whole optimizer on the CPU, the rollout vLLM with
TP2. `PRECISION` selects the arm: `mxfp4_qat` (MXFP4 rollout experts and MXFP4 QAT), `mxfp4` (MXFP4 rollout experts
only) and `bf16` (the baseline). The script reads DAPO-Math-17k in verl's format from `TRAIN_FILE`, as
[verl-recipe's `dapo/prepare_dapo_data.sh`](https://github.com/verl-project/verl-recipe/blob/main/dapo/prepare_dapo_data.sh)
downloads it, and runs without a validation set.

```bash
PRECISION=mxfp4_qat TOTAL_TRAINING_STEPS=10 bash examples/mxfp4/run_qwen3_30b_a3b_megatron.sh
PRECISION=mxfp4 TOTAL_TRAINING_STEPS=10 bash examples/mxfp4/run_qwen3_30b_a3b_megatron.sh
PRECISION=bf16 TOTAL_TRAINING_STEPS=10 bash examples/mxfp4/run_qwen3_30b_a3b_megatron.sh
```
