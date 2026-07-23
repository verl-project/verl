# TOPR

TOPR (Tapered Off-Policy REINFORCE) is a REINFORCE-family policy loss for stable off-policy updates. Positive-advantage sequences are reinforced with unit weight, while negative-advantage sequences are weighted by the sequence-level importance ratio tapered to `[0, 1]`, which bounds the update without zeroing the gradient the way PPO-style clipping does. TOPR uses no KL penalty and no reference model and reduces to REINFORCE when on-policy.

Reference: [Tapered Off-Policy REINFORCE: Stable and efficient reinforcement learning for LLMs](https://arxiv.org/abs/2503.14286).

## Canonical Scripts

| Script                  | Infer | Train | Platform |
|-------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`  | vLLM  | FSDP  | NVIDIA   |

## Key Flags

- `actor_rollout_ref.actor.policy_loss.loss_mode=topr`
- `actor_rollout_ref.actor.policy_loss.topr_negative_ratio_lower=0.0` (canonical taper lower bound)
- `actor_rollout_ref.actor.policy_loss.topr_negative_ratio_upper=1.0` (canonical taper upper bound)
- `actor_rollout_ref.actor.use_kl_loss=False` (TOPR needs no KL regularization)
- `algorithm.adv_estimator=grpo` (any outcome/group-based estimator; the advantage is constant per sequence)
