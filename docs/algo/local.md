# LOCAL: Local-Curvature Advantage Logit Regression

Last updated: 09/28/2026.

LOCAL fits the tabular policy-mirror-descent update `z_new = z_old + eta * A + c(s)` with a neural policy.
The state-dependent offset (SDO) `c(s)` leaves the target policy unchanged, but fixing it (for example
to zero) adds curvature to the fitting problem. LOCAL profiles the offset out: the optimal offset is the
old-policy mean of the logit displacement, so the regression residual is measured in the Fisher geometry
of the old policy, and its local linearisation recovers the natural policy gradient.

## Objective

For a response token `a` at prefix `s`, with `T` the `K` most probable tokens of the old policy and `pbar`
the old probabilities renormalised on `T`:

```text
psi_theta(a) = z_theta(a) - sum_{b in T} pbar(b) z_theta(b)
delta        = psi_theta(a) - psi_old(a) - eta * A
loss         = 1/2 * delta^2        (aggregated with actor.loss_agg_mode)
```

`z` are the temperature-scaled logits. With `K = |vocab|` the centre is the exact optimal SDO; the Top-K
support keeps the stored old-policy state at `K` token ids and log-probabilities per response token.
There is no ratio clipping: the gradient of a token is `delta` times the gradient of `psi_theta`, so it
vanishes once the centred logit has moved by `eta * A` and reverses if the fit overshoots.

## Implementation

- The old-log-prob pass (`compute_log_prob`) stores, per response token, the old Top-K ids
  (`old_local_topk_ids`), their log-probabilities (`old_local_topk_log_probs`) and the centred logit
  `psi_old` (`old_local_centered_logits`).
- During the actor update the FSDP engine gathers the current logits at the stored ids and returns
  `psi_theta`; `ppo_loss` computes the regression loss.
- Both the V1 trainer (`trainer.use_v1=True`, the default) and the legacy trainer are supported.

## Configuration

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: local
      local_eta: 5.0   # target scale eta
      local_topk: 64   # Top-K support size K
```

Requirements: the FSDP engine (`strategy` `fsdp` or `fsdp2`), `model.use_fused_kernels=False`, and
old-log-prob recomputation (`algorithm.rollout_correction.bypass_mode=False`). Advantages come from the
configured estimator (GRPO, GAE, ...). The actor KL loss and the entropy bonus, when enabled, are added
to the regression loss as for the other policy losses.

## Metrics

`actor/pg_loss` is the regression loss itself. `actor/local_residual_mse`, `actor/local_residual_abs`,
`actor/local_target_abs` and `actor/local_feature_delta_abs` describe the fit, `actor/local_topk_mass`
is the old-policy probability mass retained by the Top-K support, and `actor/ppo_kl` is reported for
comparison with PPO.

## Example

```bash
bash examples/local_trainer/run_qwen2_5_3b_fsdp.sh
```
