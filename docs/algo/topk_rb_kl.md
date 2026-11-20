# Top-k Rao-Blackwellised KL loss (`kl_loss_type: topk_rb`)

Last updated: 11/15/2026.

`actor_rollout_ref.actor.kl_loss_type=topk_rb` is a KL penalty against the reference policy for
`use_kl_loss=True`. It computes the per-token KL(π_θ ‖ π_ref) **exactly** over the reference's top-k tokens and
corrects for the remaining vocabulary with the sampled token. The result is an **unbiased** estimate of the
gradient of the full-vocabulary KL at every k, with much lower variance than the single-sample estimators
(`kl`/`k1`, `mse`/`k2`, `low_var_kl`/`k3`, and the `+` straight-through variants). Tracking issue:
[#8536](https://github.com/verl-project/verl/issues/8536).

## Why

The existing `kl_loss_type` options are functions of the sampled token only:

- `low_var_kl` (`k3`) has an unbiased *value*, but used as a loss its expected gradient is the gradient of the
  other direction, KL(π_ref ‖ π_θ).
- `k2` and `k3+` have the right gradient in expectation, but one sample per token makes it noisy.

Truncating to the top-k with one aggregate "tail" bucket (exact, cheap) is not a safe fix on its own. It cannot
see how probability is redistributed *inside* the tail, and a reward can move tail mass onto rewarded
low-reference-probability tokens at no KL cost.

## The estimator

Per response position let q = π_θ, p = π_ref, T the reference top-k ids, and q_t = q(∉T), p_t = p(∉T) the
tail masses. By the chain rule of KL,

$$
\mathrm{KL}(q\,\|\,p) = \underbrace{\sum_{v\in T} q_v \log\frac{q_v}{p_v} + q_t\log\frac{q_t}{p_t}}_{\text{coarse, exact}}
\;+\; q_t\,\mathrm{KL}\big(q_{\mid\text{tail}}\,\|\,p_{\mid\text{tail}}\big).
$$

The coarse term is computed exactly from the policy logits. For the within-tail term, with the sampled token
y and r = log(q_y / q_t) − log(p_y / p_t), the surrogate

$$
\mathbb{1}[y\notin T]\,\big(\operatorname{sg}(r)\,\log q_t + \tfrac12 r^2\big)
$$

has expected gradient equal to the gradient of the within-tail term when y ~ q. Training tokens are drawn by
the rollout policy, so the term is weighted by the **detached** ratios π_θ / π_old (PPO mini-batch reuse; 1 on
the first mini-batch) and `rollout_is_weights` = π_old / π_rollout when rollout correction is enabled. The
reported `kl_loss` metric is the matching unbiased value estimate.

- k only trades cost for variance. With T = the whole vocabulary, it is the exact KL and its exact gradient.
- This is the per-token Rao-Blackwellised KL of
  [Amini, Vieira & Cotterell, NeurIPS 2025](https://arxiv.org/abs/2504.10637), restricted to the top-k, plus
  the sampled tail correction. [TIKE (ACL 2026)](https://aclanthology.org/2026.acl-long.405/) truncates to
  the top-k without that correction.
- The tail masses are log-sum-exps over the complement of T, taken relative to the tail's own maximum, not
  1 − Σ top-k. They stay exact when the reference's top-k holds nearly all of its mass (even hundreds of nats
  below the top token in fp32), which is exactly where a drifted policy's tail mass matters most.
- Padded LM-head columns are treated like any other token. The top-k, the tails and the sampled-token
  log-probs all use the same softmax, so the estimate is unbiased for the KL of the distribution the model
  actually defines.

## Usage

```yaml
actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_type: topk_rb
    kl_topk: 16          # reference top-k size
    kl_loss_coef: 0.001
```

How it is wired:

- The reference forward (`compute_ref_log_prob`, or the actor with the LoRA adapter disabled when
  `ref_in_actor`) also emits `ref_topk_ids` and `ref_topk_log_probs` for every token. The log-probs have k + 1
  columns: the last one is the log of the reference's tail mass.
- During the actor update the engine computes the coarse KL and the policy's log tail mass from the full
  logits. The top-k log-probs reuse `topk_log_probs_from_logits` (the chunked, vocab-parallel function of score
  centering); the exact tail mass is a small autograd function written the same way. `ppo_loss` adds the
  sampled tail term.
- Metrics: `kl_loss` (the unbiased KL estimate), `kl_topk/policy_tail_mass` and `kl_topk/ref_tail_mass`
  (the mass outside the reference top-k).
- Supported: FSDP/FSDP2 (including Ulysses SP), VeOmni, and Megatron (vocab-parallel TP, CP).

Rejected at startup:

- other strategies;
- `model.use_fused_kernels=True` (the full logits are never materialised);
- `algorithm.kl_penalty=topk_rb` (KL in reward), since this is a loss-side estimator;
- `pad_to_length` with FSDP or VeOmni (Megatron packs the reference top-k like its labels and supports it).

## Cost

| | per token | example: 2,048 sequences × 4,096 tokens |
|---|---|---|
| ref → actor payload, k = 16 | 132 B ((k + 1) fp32 log-probs + k int32 ids) | ≈ 1.1 GB |
| ref → actor payload, k = 128 | 1 KB | ≈ 8.6 GB |

- The payload covers **prompt and response** tokens, because the actor packs it alongside the full token
  stream. Budget it with the rollout batch, and keep k small; k = 16 is the default.
- On the actor, the coarse KL saves only a reference to the logits and recomputes the softmax chunk by chunk in
  backward. As for score centering and entropy, the log-prob cross entropy must then leave the logits intact:
  on FSDP its in-place backward is turned off (one extra logits-sized gradient buffer per micro-batch), and on
  Megatron it is given a clone of the `(tokens, vocab / tp)` logits.

## Measurements

On real model distributions, π_θ = Qwen2.5-0.5B-Instruct and π_ref = Qwen2.5-0.5B. The setup was 20 sampled
responses (2,560 positions), with estimator statistics over 500 positions computed by exact expectation over
the sampled token:

| | k = 4 | k = 16 | k = 64 | k = 128 |
|---|---|---|---|---|
| reference top-k mass P, median / 10th pct | 0.952 / 0.429 | 0.991 / 0.652 | 0.998 / 0.813 | 0.999 / 0.870 |
| gradient RMSE vs `k2`/`k3+`, median per position | 0.65 | 0.33 | 0.18 | 0.14 |
| positions where `topk_rb` beats `k2` | 97% | 99.6% | 100% | 100% |
| relative gradient bias (median) | 1e-14 | 1e-14 | 2e-14 | 2e-14 |

In the same setup the `k3` gradient has a median relative bias of 0.46, which is the direction effect above.

In a tabular GRPO problem with a closed-form KL-regularised optimum, the gap to the optimal objective was
0.005–0.009 with `topk_rb` at k = 5…100. For comparison it was 0.003 with the exact full KL, 0.008 with `k3+`
and 0.56 with `k3`. The truncated coarse KL without the tail term ended farther from π_ref than `k3`.

## Tests

- `tests/trainer/ppo/test_topk_rb_kl_on_cpu.py`:
  - exactness at k = V;
  - agreement with a plain-autograd oracle, values and gradients, single device and vocab-parallel (gloo);
  - unbiasedness of gradient and value by exact expectation, on- and off-policy, with controls showing each
    importance weight is needed;
  - the variance advantage over `k2`;
  - padded columns, masked (`-inf`) logits, and fp32 tails far below the top token.
- `tests/workers/test_topk_rb_kl_plumbing_on_cpu.py`:
  - FSDP engine outputs in both padding modes;
  - the left-right padding round trip;
  - an end-to-end check from ref engine to `ppo_loss`, against the estimator on directly aligned tensors;
  - startup validation.
- `tests/workers/test_topk_rb_kl_megatron_on_cpu.py`: the Megatron logits processor under real TP groups
  (gloo), compared with single-device autograd.
- `tests/trainer/ppo/test_topk_rb_kl_trainer_wiring_on_cpu.py`: the trainer requests, stores and forwards the
  reference top-k.
