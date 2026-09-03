# W4A4 with truncated importance sampling off

Single-variable arm against the v13 long run
(`verl_30b_realw4a4_r3_nativeonline_8n_long_20260901_v13`). Only `ROLLOUT_IS=null`
differs; precision, R3, 0/3 loss, clip, lr, batch, dataset and the v12 runtime are
identical.

## Hypothesis

verl weights every token's policy loss by `w = clamp(exp(old_logp - rollout_logp), max=2)`.
The clamp fraction is small (0.11% of tokens high, 0.28% low), which is why this was
initially dismissed - but the clamp fraction is the wrong statistic. The weight applies to
**every** token, and its spread is precision-dependent:

| arm | `rollout_is_std` | `rollout_corr/kl` | length |
| --- | --- | --- | --- |
| verl BF16 0603 | 0.037 | 0.0012 | grows (+34.6 tok/step to step 265) |
| **verl W4A4 v13** | **0.089** | **0.0044** | **flat (+0.88 over 260 steps)** |
| NeMo-RL W4A4 (no IS correction at all) | n/a | n/a | grows (+16.9 tok/step) |

W4A4 train-rollout mismatch is tail-dominated (an earlier finding in this workstream:
the bottom 20% of tokens by `rollout_logp` carry ~74% of total |delta|). If mismatch
accumulates with token position, TIS systematically down-weights later tokens, which
suppresses exactly the thing that is not growing.

This also explains why BF16 and NeMo both grow: BF16 keeps TIS but its mismatch is ~4x
smaller so `w ~ 1` and TIS is inert, and NeMo's `loss_fn` carries no importance-sampling
correction at all (`ratio_clip_min/max/c` = 0.2/0.28/10 match verl exactly, and its
token-level loss uses the same `sum(loss*mask)/global_valid_toks` normalization).

## Ruled out before getting here

Implementation (all attestations pass, KL 0.0044, ESS 0.993, 18432 expert weights,
`changed=1` on 98/101 refits), truncation (`max` hits 20480, `clip_ratio` ~0), entropy
(BF16 is *lower* at 0.086 vs 0.122 and still grows), FP4 itself (NeMo reaches 4027 on the
same precision contract), dynamic sampling (~4% degenerate groups), loss normalization
(identical), clip parameters (identical), and generation quality (step-259 samples are
coherent, correctly-scored CoT).

## Reading the result

- **Length takes off** -> TIS is the length suppressor under real W4A4. The fix is a
  precision-aware IS policy, not a length or entropy knob.
- **Still flat** -> TIS is exonerated too, and the cause is in verl's W4A4-specific
  numerics; next step is the BF16 control, which needs verl's BF16 weight sync ported to
  vLLM 0.26's RoutedExperts API.

Watch `rollout_corr/kl` and `actor/entropy` in the first 3 steps: a broken arm shows
entropy ~6 and KL > 1, and must be killed rather than left to run.

```bash
./submit.sh probe && ./submit.sh build && ./submit.sh preflight
./submit.sh smoke && ./submit.sh control
```
