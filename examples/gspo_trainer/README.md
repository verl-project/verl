# GSPO (Group Sequence Policy Optimization)

GSPO is a GRPO-family policy-loss variant that aggregates the IS-ratio at the sequence level (`seq-mean-token-mean`) and uses a very tight clip window. It is especially useful for large MoE models.

Reference: [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071).

## Canonical Scripts

| Script                                      | Infer | Train    | Platform |
|---------------------------------------------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`                 | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_fsdp.sh`             | vLLM  | FSDP     | Ascend   |
| `run_qwen3_30b_a3b_megatron.sh`        | vLLM  | Megatron | NVIDIA   |

## Key Flags

- `actor_rollout_ref.actor.policy_loss.loss_mode=gspo`
- `actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean`
- `actor_rollout_ref.actor.clip_ratio_low=3e-4`
- `actor_rollout_ref.actor.clip_ratio_high=4e-4`
- `actor_rollout_ref.actor.clip_ratio_c=10.0` (dual-clip guard)
