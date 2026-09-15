# Real W4A4 NVFP4 with Megatron and vLLM

This directory implements end-to-end real W4A4 independently of Verl's legacy
QAT path.

## Precision contract

- Training uses Megatron-Core plus the official Transformer Engine `2.18.0`
  release (`transformer-engine`, `transformer-engine-cu13`, and
  `transformer-engine-torch` must all match). The release contains the
  GroupedLinear packed-wgrad and dequantized-backward operand fixes. Persistent policy
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
training PP=1 with no virtual pipeline, BF16 KV cache, and no speculative decoding.
Mixed dense/MoE layouts are rejected: the current training module recipe does
not preserve BF16 for their dense MLPs. An evaluation recipe must be absent or
identical to its training recipe, including BF16 carve-outs.

An unmodified vLLM 0.26 wheel is **not** sufficient. Apply the audited fixes
using `runtime_backports/apply_vllm_online_nvfp4_50029_50074.py` in the runtime
build: they remove an extra BF16 rounding step and preserve the MoE kernel
across refits. The script also includes a local derived-scale lifecycle fix:
post-processing must use the newly loaded tensors, while the retained execution
kernel keeps references to the original storage that native reload updates in
place. Applying kernel reuse alone leaves TRT-LLM's derived `g1_scale_c` one
refit behind and can split eager and CUDA-graph references. The worker verifies
both scale references and current derived values after loading/refit.
The reciprocal activation scales are registered on the layer as well: level-2
sleep discards their allocation, so keeping only a non-parameter quant-config
reference would leave them uninitialized after wake-up. Native copyback now
restores those original addresses along with the packed weights and derived
scales. Runtime guards check both their references and reciprocal values.
The worker checks the normalized source of these implementations;
an unpatched or changed implementation fails before training. Updating that
allowlist requires re-auditing the dependency, not adding a version marker.
Refit verifies unique coverage of every `(layer, expert, projection)` key as
well as native W4A4 layer counts. These checks do not replace GPU graph/eager
equivalence tests after consecutive reloads.

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

## Historical scheduler workflow

The versioned `r3_nativeonline_20260831_v3` scripts below reproduce an older
runtime, **not** the current TE 2.18 release build. Do not use their source-pin
build step to validate the current lockfile. Current validation must use a
fresh versioned image and frozen checkout, preserving the eight-node recipe
and explicit runtime backports. Run tests through the cluster scheduler, not
on a shared login node. Independent formal training chains need not wait for
unrelated user jobs; test jobs should remain low-concurrency.

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

## Dynamic sampling and diagnostic boundaries

Generation batch size and policy-update batch size are distinct. The launcher
defaults above are not the overlong/dynamic-sampling experiment contract;
record the resolved `FILTER_GROUPS`, `GEN_PROMPT_BSZ_MULT`, overlong settings,
and first/last BF16 carve-outs for every comparison.

With dynamic sampling, optimizer steps do not count consumed dataloader
batches. New checkpoints store separate dataloader progress alongside
`data.pt`. Resuming a legacy dynamic-sampling checkpoint without that progress
requires the explicit zero-based `trainer.dataloader_resume_epoch`; it cannot
be recovered reliably from the optimizer step. Preserve full Adam state too.

Paired log-prob dumps require a recomputed actor forward and reject bypass
mode. They compare actor and rollout on each arm's own generated trajectories,
not four evaluations on one shared set of answers. Refills are included in
pre-filter token counts and request-latency summaries; these summaries do not
measure active GPU concurrency. Passing the guards is not proof of quality
parity or end-to-end speedup.
