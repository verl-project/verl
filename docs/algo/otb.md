# Optimal Token Baseline (OTB)

Last updated: 12/25/2025.

Optimal Token Baseline (OTB) is dynamic token-level baseline for variance reduction. It weights updates based on "Realized Energy"—essentially, how much uncertainty has accumulated up to that specific token. It downweights the noisy parts and trusts the clear signals. Read [Optimal Token Baseline blog](https://richardli.xyz/optimal-token-baseline) for more details.

## The method: OTB

- OTB builds a _dynamic_ baseline that adapts to each token by tracking the “Realized Energy”—the uncertainty that has accumulated up to that token. It downweights the noisy parts and trusts the clear signals.
- Unlike standard group means (which average over the padding `EOS` token ineffectively), OTB handles this naturally by computing baselines only over valid tokens.

## Logit-Gradient Proxy

- Computing true uncertainty requires expensive backward passes (calculating gradient norms per token). Instead, OTB introduces the **Logit-Gradient Proxy**: the realized energy can be estimated entirely from forward probabilities.
- This means zero extra backward calls and effectively no additional runtime overhead.

## Mechanics at a glance

For each prompt group of size `N`, OTB computes rewards-to-go `G_t` and cumulative variance weights `W_t`. The optimal baseline per token is

```
B*_t = (Σ_i G_t^{(i)} · W_t^{(i)}) / (Σ_i W_t^{(i)} + ε),
W_t = Σ_{j=1}^t (1 - 2π_j + Σπ_j²),
Σπ_j² = exp(logsumexp(2·logits_j) - 2·logsumexp(logits_j)).
```

The final advantage is `(G_t - B*_t) · mask_t`, so padding tokens stay at zero.

## Integration in VERL

- `AdvantageEstimator.OPTIMAL_TOKEN_BASELINE` registers `compute_optimal_token_baseline_advantage`, invoked whenever `algorithm.adv_estimator` is set to `optimal_token_baseline`.
- `ActorRolloutRefWorker.compute_log_prob` emits an additional tensor `sum_pi_squared` (Σπ² per token) when `actor.calculate_sum_pi_squared=True`. This requires disabling fused log-prob kernels, because they do not surface logits.
- Trainers assert `sum_pi_squared` exists, regroup trajectories by `non_tensor_batch["uid"]`, and run the OTB calculation. If rollout IS is active, they rescale the weights by `rollout_is_weights**2` before aggregating.
- In Ulysses sequence-parallel setups, the actor gathers, unpads, and returns Σπ² in the same way it handles log-probabilities, so OTB supports sharded sequence-parallel models out of the box.
- `sum_pi_squared_checkpointing` is available to trade compute for memory when Σπ² tensors become large (e.g., lengthy chain-of-thought reasoning).

## Configuration checklist

- `actor_rollout_ref.actor.calculate_sum_pi_squared: true` (mandatory).
- `actor_rollout_ref.model.use_fused_kernels: false` (required until fused kernels emit logits).
- `algorithm.adv_estimator: optimal_token_baseline`.
- Group sampling (`actor_rollout_ref.rollout.n > 1`) to unlock OTB’s variance reduction; with `n=1` the baseline collapses to returns.

Example OmegaConf overlay:

```yaml
algorithm:
  adv_estimator: optimal_token_baseline

actor_rollout_ref:
  actor:
    calculate_sum_pi_squared: true
    sum_pi_squared_checkpointing: false # optional memory saver
  rollout:
    n: 8
```

## Example script

- `examples/otb_trainer/run_qwen2_5-7b.sh`.

## Gradient Variance Proxy Metrics

All gradient-variance analysis in the Optimal Token Baseline work starts from the variance identity

```
Var(ĝ) = E[||ĝ||²] - ||E[ĝ]||²,
```

which states that the variance of any stochastic gradient equals the mean squared magnitude minus the squared norm of its expectation.

For a trajectory `τ`, the policy-gradient estimator is

```
ĝ(τ) = ∇ log π_θ(τ) · A(τ),        A(τ) = R(τ) - B.
```

The logit-gradient proxy approximates the squared gradient norm without an extra backward pass:

```
||ĝ(τ)||² ≈ Ŵ(τ) · A(τ)²,
```

where `Ŵ(τ)` is the realized energy built. Given a mini-batch `{τ_i}` of size `N`, we decompose its statistics into three diagnostics:

- **Signal strength (squared norm of the mean gradient)**
  ```
  S = || (1/N) · Σ ĝ(τ_i) ||²
  ```
- **Total power (signal + noise)**
  ```
  P_total = (1/N) · Σ Ŵ(τ_i) · A(τ_i)²
  ```
- **Pure noise (estimated variance of the batch mean)**
  ```
  Var_proxy = (1/(N-1)) · (P_total - S)
  ```

`verl/trainer/ppo/metric_utils.py#L306` implements these diagnostics via `compute_variance_proxy_metrics`, emitting
`variance_proxy/proxy1_signal_strength`,
`variance_proxy/proxy2_total_power`, and
`variance_proxy/proxy3_pure_noise`.

Tracking these metrics provides a forward-only, low-overhead view of gradient health for any advantage estimator that supplies `sum_pi_squared`.
