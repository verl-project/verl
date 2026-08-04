# Direct Reward Optimization

Last updated: 07/29/2026.

Select `actor_rollout_ref.actor.policy_loss.loss_mode=dro` and set a positive
`actor_rollout_ref.actor.policy_loss.dro_beta`. The token loss is

$$
L_{\mathrm{DRO}} = -\sum_t \left[\log \pi_\theta(a_t\mid s_t) A_t
- \frac{\beta}{2}\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t)\right)^2\right].
$$

DRO expects the caller to construct advantages appropriate for its soft off-policy formulation.
The objective honors `response_mask`, global loss aggregation, and optional rollout-correction
weights in the same way as the other registered policy losses.
