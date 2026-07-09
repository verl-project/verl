# TOPR: Tapered Off-Policy REINFORCE

Last updated: 07/08/2026.

Tapered Off-Policy REINFORCE (TOPR) is a REINFORCE-family objective for stable off-policy updates. It treats sequences asymmetrically by advantage sign: positive-advantage sequences are reinforced with unit weight (no importance correction), while negative-advantage sequences are weighted by the sequence-level importance ratio tapered to `[0, 1]`. The tapered weight bounds the update on off-policy negative sequences without zeroing their gradient the way PPO-style clipping does, so the policy keeps learning to suppress bad sequences even when they have drifted off-policy. TOPR uses no KL penalty and no reference model, and it reduces to plain REINFORCE when on-policy. For more details, please refer to the original paper [Tapered Off-Policy REINFORCE: Stable and efficient reinforcement learning for LLMs](https://arxiv.org/abs/2503.14286).

## Key Components

- Asymmetric taper: unit weight for positive-advantage sequences; `clip(ratio, 0, 1)` importance weight for negative-advantage sequences, where the ratio is the length-normalized sequence-level importance ratio between the current and behavior policies.
- The taper weight is a constant (stop-gradient), so the gradient is the tapered REINFORCE gradient.
- No KL penalty, no reference model, no critic required.
- Trajectory-level objective: intended for outcome/group-based advantage estimators (e.g. `grpo`, `rloo`, `reinforce_plus_plus`) where the advantage is constant within a sequence.

## Configuration

To configure TOPR within the framework, use the following YAML settings.

```yaml
algorithm:
  adv_estimator: grpo  # any outcome/group-based estimator
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "topr"
```

The taper bounds applied to negative-advantage sequences default to the canonical `[0, 1]` and can be adjusted:

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "topr"
      topr_negative_ratio_lower: 0.0
      topr_negative_ratio_upper: 1.0
```
