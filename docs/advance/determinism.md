# Full Determinism for Reproducible RL Training

**Authors**: Haichuan Hu, Yongxiang Huang, Jiawei Zhang, Nguyen Long

Last updated: 11/12/2026.

The end-to-end reproducibility workflow requires the legacy V0 trainer, which has
been removed from mainline. For its configuration and validation commands, use
the [verl v0.9.1 guide](https://github.com/verl-project/verl/blob/v0.9.1/docs/advance/determinism.md)
with that version of verl.

Mainline always uses V1. `trainer.use_v1` is retained for script compatibility;
setting it to `false` does not select V0. Engine-level `full_determinism` settings
do not guarantee end-to-end reproducibility with V1: rollout collection order can
vary across runs.

## Batch-invariant actor


`actor_rollout_ref.actor.fsdp_config.batch_invariant=true` puts the actor forward under the same batch-invariant setup a vLLM rollout gets from `VLLM_BATCH_INVARIANT=1`, and computes the actor's log-probs with the vLLM sampler formula: fp32 logits divided by the fp32 temperature, fp32 `log_softmax` over the full vocabulary, then gather. This is the training-side half of exact 0-diff train-rollout log-probs: an actor forward built from the same kernels as the rollout can then be compared bitwise with `rollout_log_probs`, using `training/rollout_logprobs_mismatch_count`; `actor/old_logprobs_mismatch_count` counts the response tokens whose update-forward log-prob differs from `old_log_probs`, so the first update of a step can be checked the same way (a mean such as `actor/ppo_kl` can be zero while tokens differ). It does not by itself make the actor match the rollout; the model's remaining operators (attention, normalization, rope, MoE) still have to be aligned per model. Requires vLLM in the training process. FSDP engine only; `use_fused_kernels` is rejected because the fused kernel computes its own log-probs. Entropy and `sum_pi_squared` keep their usual kernels; the 0-diff target is the log-probs the loss uses. The rollout reports `processed_logprobs`, taken after top-k/top-p masking, so the target also needs `rollout.top_k=-1` and `rollout.top_p=1`.

It works in two places:
- The entrypoint (`main_ppo`) exports, when any FSDP engine config (actor, ref or critic) sets `batch_invariant`, `VLLM_BATCH_INVARIANT=1`, `CUBLAS_WORKSPACE_CONFIG=:16:8`, `CUBLASLT_WORKSPACE_SIZE=1` and vLLM's two compile switches before `ray.init()`, and forwards them to all Ray actors. cuBLAS reads its workspace config when it initializes, so it cannot be set from inside the worker. On SM90/SM100 vLLM registers no matmul override: matmul invariance comes only from this cuBLAS workspace setting and cuBLASLt selection, which is why the env must be exported this early. Because `VLLM_BATCH_INVARIANT=1` reaches every worker, the rollout also runs batch-invariant, and the rollout process applies vLLM's NCCL profile itself; that profile is not forwarded to the FSDP workers, where it would throttle the training collectives.
- The FSDP engine calls `init_batch_invariance()` in the worker to install the aten overrides (softmax, log_softmax, mean, bmm, and on SM80 matmul) and the TF32/cuBLASLt settings, fails if vLLM skipped the install, then reverts the environment vLLM wrote and writes the entrypoint's values back: the training communicators are created after this point, and vLLM's NCCL profile would throttle every FSDP collective and the weight sync. vLLM 0.29-0.31 also replace `torch.bmm` itself with a kernel launch that drops autograd; the engine restores it, and bmm stays batch-invariant through the aten override. The install is process-global: actor and ref share a worker, so the ref forward, the backward and the optimizer step all run on the overrides. On SM80 that routes every GEMM through Triton, which costs step time; on SM90/SM100 the overrides cover softmax, log_softmax, mean and bmm only.

`batch_invariant` does not touch the NCCL settings; `full_determinism` keeps owning them. `CUBLAS_WORKSPACE_CONFIG=:16:8` is the same under both.
