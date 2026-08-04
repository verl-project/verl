# Importance Sampling

Last updated: 07/29/2026.

Select `actor_rollout_ref.actor.policy_loss.loss_mode=importance_sampling` to optimize

$$
L_{\mathrm{IS}}
=
-\sum_t
\frac{\pi_\theta(a_t \mid s_t)}
     {\pi_{\mathrm{old}}(a_t \mid s_t)}
A_t
$$

The objective is unclipped. Use `actor_rollout_ref.actor.loss_agg_mode` to choose its reduction.

The objective honors `response_mask`, global loss aggregation, and optional rollout-correction
weights in the same way as the other registered policy losses.
