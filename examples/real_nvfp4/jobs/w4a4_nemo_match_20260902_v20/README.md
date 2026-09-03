# W4A4 matched to NeMo RL's actual DAPO recipe

Two knobs against the v13 long run, because they are one mechanism:
**strict Minerva scoring + DAPO dynamic sampling**.

## Why both, and why v19 failed with only one

| arm | verifier | dynamic sampling | reward | length |
| --- | --- | --- | --- | --- |
| v13 | lenient (`\boxed{}` fallback) | off | −0.637 | flat 1015 → 1124 |
| **v19** | **strict** | **off** | **−0.914 (matches NeMo)** | **collapsed 1015 → 488 @23** |
| NeMo | strict | **on** | −0.875…−0.938 | grows → 4027 |

v19 confirmed the scoring half: its reward landed exactly in NeMo's early band.
But length collapsed, and the reason is visible in NeMo's own logs:

```
NeMo step 1-12: reward −0.875…−0.938 (acc 3-7%),  all-wrong groups = 0.00%
```

At ~6% per-sample accuracy with 16 samples the naive all-wrong rate is
`0.94^16 ≈ 37%`. NeMo reports **0.00%** because `use_dynamic_sampling=True`
(`batch_multiplier=3`, `dynamic_sampling_max_gen_batches=20`) resamples until the
batch contains no degenerate groups.

Strict scoring puts verl in that same low-accuracy regime. Without resampling,
roughly half the prompt groups are all-wrong, GRPO gives them zero advantage, and
the gradient is gutted — so the model drifts toward short outputs instead of
learning the format.

**Correction to an earlier conclusion in this investigation:** dynamic sampling
was dismissed on the grounds that NeMo's degenerate-group rate is only ~4%. That
figure is the rate *after* resampling has already filtered — a residual, not the
underlying rate. Reading a post-filter statistic as pre-filter understated the
mechanism by an order of magnitude.

## Settings

`VERL_MATH_DAPO_STRICT_MINERVA=1` (via Ray `runtime_env`, since the scorer runs in
workers), `FILTER_GROUPS=True`, `MAX_GEN_BATCHES=20`, `GEN_PROMPT_BSZ_MULT=3`
(gen batch 96 vs train 32). Everything else identical to v13: precision, R3,
0/3 loss, clip, lr, TIS default, dataset, v12-lineage runtime.

## Reading the result

- **Length takes off** → the plateau was a recipe mismatch (scoring + sampling),
  not W4A4. Pin both in the formal recipe and re-run the long chain; v13's 260
  steps are not comparable to NeMo/slime.
- **Still flat or collapsing** → recipe is not sufficient; reopen the W4A4
  numerics line and unblock the BF16 control.

Gates: step-1 `critic/score/mean` should sit near −0.9 (strict active); watch that
length does **not** repeat v19's collapse; kill on entropy > 3 or KL > 1.

```bash
./submit.sh probe && ./submit.sh build && ./submit.sh preflight && ./submit.sh control
```
