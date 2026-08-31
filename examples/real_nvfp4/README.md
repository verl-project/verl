# Real W4A4 NVFP4 with Megatron and vLLM

This directory implements end-to-end real W4A4 independently of Verl's legacy
QAT path.

## Precision contract

- Training uses Megatron-Core plus the exact Transformer Engine
  `2.18.0+e7c550c5` source pin from the current NeMo RL PR #3566. This pin
  contains the GroupedLinear packed-wgrad and dequantized-backward operand
  fixes. Persistent policy
  parameters remain BF16 (`fp4_param=false`) and the optimizer keeps its full
  Adam state.
- The per-module TE recipe keeps attention BF16 and applies NVFP4 to every
  routed-expert MLP `linear_fc1`/`linear_fc2`. The backward path is currently
  `dequantized`.
- TE adaptive 4-over-6 is deliberately off: vLLM 0.26's native
  `nvfp4_per_token` rollout consumes standard NVFP4, so enabling 4-over-6 only
  on the training side would create a precision mismatch.
- Actor-to-rollout refit transports ordinary BF16 checkpoint tensors. It never
  sends actor-side packed FP4 weights or activation scales.
- Each vLLM worker uses vLLM 0.26's built-in `nvfp4_per_token` online method
  while native `model_runner.reload_weights(..., is_checkpoint_format=True)`
  consumes the stream. It batches complete expert layers before quantization;
  the fused gate/up matrix shares one global scale per expert and down uses its
  own.
- Rollout uses the FlashInfer TRT-LLM NVFP4 fused-MoE kernel with
  `per_token_activation=True`. Attention, router, norms, embeddings, shared
  experts, and the LM head stay BF16.
- CUDA Graphs remain enabled with `FULL_DECODE_ONLY`. The final IPC ACK is sent
  only after native layerwise reload, post-processing, attestation, and a device
  fence complete.

The runtime is intentionally fail-closed for the validated
`Qwen3MoeForCausalLM` all-MoE layout, vLLM 0.26.0, rollout TP/PP/EP = 1/1/1,
BF16 KV cache, and no speculative decoding. It verifies exact expert tensor and
native W4A4 layer counts rather than accepting a partial refit.

## R3 and loss contract

The formal recipe enables R3 on both sides. vLLM 0.26 skips
`router.select_experts` in monolithic fused-MoE kernels, so this integration
installs the routed-expert capture hook inside every vLLM model worker and
rejects an all-zero route payload.

This version contains none of the three Slime stability losses. It uses the
plain token-mean PPO/GRPO loss, token-level TIS, no KL loss/reward, and no DAPO
overlong penalty. Formal scale is 8 nodes × 4 GPUs, EP=4, 32 prompts × 16
responses, and `max_num_seqs=128`.

## Entry points

- Training recipe: `run_qwen3_30b_megatron.sh`
- TE module recipe: `config/attn_bf16_mlp_nvfp4.yaml`
- vLLM quantization scope: native `nvfp4_per_token` (MoE only; linears BF16)
- Versioned scheduler bundle: `jobs/r3_nativeonline_20260831_v3/`

## Gated scheduler workflow

Run exactly one phase at a time. Each phase writes a versioned `.pass` marker;
the next phase refuses to start without it, and `submit.sh` refuses to add work
while this user already has a running or pending job:

```bash
bash examples/real_nvfp4/jobs/r3_nativeonline_20260831_v3/submit.sh probe
bash examples/real_nvfp4/jobs/r3_nativeonline_20260831_v3/submit.sh build
bash examples/real_nvfp4/jobs/r3_nativeonline_20260831_v3/submit.sh preflight
bash examples/real_nvfp4/jobs/r3_nativeonline_20260831_v3/submit.sh smoke
bash examples/real_nvfp4/jobs/r3_nativeonline_20260831_v3/submit.sh short
```

`probe` inspects the old base without modifying it. `build` creates a new,
checksummed aarch64 image with vLLM 0.26 and the exact TE source pin.
`preflight` is the only place that
runs unit/GPU checks. `smoke` performs two real updates on one node; only then
can the eight-node short arm be submitted. Logs and phase state live under the
versioned `ray_log/` and `run_state/` directories next to the worktree.
Smoke and short-run experiment IDs include the Slurm job ID, so a failed
compatibility attempt cannot collide with a corrected retry or accidentally
resume its W&B/checkpoint state.

For manual development outside this production gate, run a fresh arm with a
unique experiment ID:

```bash
PRECISION_MODE=real_nvfp4 EXP_NAME=my-new-run \
  bash examples/real_nvfp4/run_qwen3_30b_megatron.sh
```

Set `PRECISION_MODE=bf16` for the matched control. Do not resume the earlier
W4A4 checkpoints: they used actor-side packing, static rollout activation
semantics, and a different quantization scope, so they do not validate this
implementation.
