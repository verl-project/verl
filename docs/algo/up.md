# UP-GRPO: Unbounded Positive Asymmetric Optimization

Last updated: 07/12/2026.

UP (Unbounded Positive) is a plug-and-play modification of the GRPO policy loss that targets the **exploration-stability dilemma** in RL for LLMs. Standard GRPO/PPO uses a symmetric clip (single `ε`) together with importance sampling against the old policy `π_old`. For rare, low-confidence but *correct* tokens (large positive advantage, small probability), the symmetric upper clip and the `π_old` anchor suppress exactly the gradient signal that would encourage the model to explore, while lifting the clip risks importance-sampling induced gradient explosion. UP resolves this by treating the two signs of the advantage asymmetrically.

For more details, refer to the paper [UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma](https://arxiv.org/pdf/2607.06987).

## Key Components

- **Probability Capacity view**: the improvement a token can contribute is bounded by how much probability mass it can still gain. Low-probability correct tokens have the most capacity, yet the symmetric clip caps their update the hardest. UP is designed to unlock this capacity.
- **Unbounded positive**: for positive advantages, UP removes both the clip and the `π_old` importance-sampling anchor, so low-confidence correct tokens are free to grow.
- **Asymmetric design**: positive and negative advantages are optimized with different objectives — unbounded exploration on the positive side, conservative clipping on the negative side for stability.

## The Objective

UP-GRPO uses the following token-level objective (paper Eq. 15), applied per token `(i, t)`:

- **Positive advantages (Â > 0)**: an *unbounded, self-anchored* REINFORCE objective

  \[
  \hat{A}_{i,t} \cdot \log \pi_\theta(o_{i,t}\mid q, o_{i,<t})
  \]

  This is implemented with the stop-gradient trick: the self-anchored ratio `r̃ = πθ / sg(πθ)` equals `1` in value but has gradient `∇ log πθ`. There is no `π_old` term and no upper clip, so the effective update reduces to REINFORCE and maximizes exploration.

- **Negative advantages (Â ≤ 0)**: the standard GRPO symmetric clip plus a dual-clip safeguard

  \[
  \min\!\big(r\,\hat{A},\; \operatorname{clip}(r, 1-\varepsilon, 1+\varepsilon)\,\hat{A}\big)
  \]

  with the dual-clip lower bound (`clip_ratio_c`) retained as a stability guard against destructive large-ratio negative updates.

The KL regularization term `β D_KL` is **not** part of this loss. It is handled separately by verl through `use_kl_loss` / `kl_loss_coef` (KL loss added to the actor loss), consistent with GRPO.

## Configuration

To enable UP-GRPO, set the policy loss mode to `up` and keep the standard GRPO advantage estimator and KL-loss configuration:

```yaml
algorithm:
  adv_estimator: grpo
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "up"
    clip_ratio: 0.2
    use_kl_loss: True
    kl_loss_coef: 0.001
    loss_agg_mode: "token-mean"
```

Notes:

- `actor_rollout_ref.actor.policy_loss.loss_mode=up`: selects the UP-GRPO policy loss.
- `algorithm.adv_estimator=grpo`: UP-GRPO builds on GRPO group-relative advantages.
- `actor_rollout_ref.actor.clip_ratio=0.2`: the clip range. It only affects **negative** advantages; for positive advantages the upper clip is removed entirely, so raising or lowering it does not change the positive-advantage update.
- `actor_rollout_ref.actor.use_kl_loss=True` and `kl_loss_coef=0.001`: KL regularization is applied by verl outside the policy loss.
- `actor_rollout_ref.actor.loss_agg_mode=token-mean`: token-level aggregation, matching the paper.

## Reference Example

```bash
bash examples/grpo_trainer/run_up_grpo_qwen3_8b_fsdp.sh
```

## Reference

UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma. Chongyu Fan et al., ByteDance Seed & Michigan State University. arXiv:2607.06987. Project page: <https://chongyu-fan.netlify.app/posts/up/>.
