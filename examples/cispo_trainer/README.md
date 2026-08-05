# CISPO

CISPO (Clipped IS-weight Policy Optimization) is a policy-loss variant that decouples the lower/upper clip ratios to stabilize IS-ratio-weighted updates, used in MiniMax-M1.

Reference: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585).

## Canonical Scripts

| Script                               | Infer | Train | Platform |
|--------------------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`          | vLLM  | FSDP  | NVIDIA   |

## Key Flags

- `actor_rollout_ref.actor.policy_loss.loss_mode=cispo`
- `actor_rollout_ref.actor.clip_ratio_low=10` (effectively unclamped on lower side)
- `actor_rollout_ref.actor.clip_ratio_high=0.2`
