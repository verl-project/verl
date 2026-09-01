# BF16 reference curve: `gb200_30B_bf16_megatron_0603`

The comparison baseline for the real-NVFP4 long chain. Recorded 2026-09-01 so the
target numbers do not have to be re-derived every time.

## How to read it in W&B

The reference is **not one run**. It is 12 runs that all share the display name
`gb200_30B_bf16_megatron_0603`, one per 5-hour Slurm segment, each resuming from
the previous segment's checkpoint. One of them (`xaduxf0a`) has no history.
Their `_step` counters are per-run and are *not* the global training step, so
stitch by concatenating segments in `created_at` order and using the cumulative
row index.

| segment | run id | training rows |
| --- | --- | --- |
| 1 | `7a6qxjqp` | 112 |
| 2 | `6bgjqwzb` | 79 |
| 3 | `7xommzdu` | 74 |
| 4 | `7fs8133e` | 67 |
| 5 | `ilz3hbo1` | 62 |
| 6 | `oe8mnow8` | 56 |
| 7 | `lyt568dw` | 52 |
| 8 | `izol4n5m` | 48 |
| 9 | `jpjh7d07` | 51 |
| 10 | `yv8l3cka` | 47 |
| 11 | `ipm3jj1o` | 43 |

Total 691 training steps. Segment length shrinks because step time grows with
response length - the same reason the v13 chain uses shrinking step targets.

## Curve

| metric | step 1 | mean of first 20 | step 265 | step 691 | slope over steps <= 265 |
| --- | --- | --- | --- | --- | --- |
| `response_length/mean` | 759.1 | 827.3 | 7712.6 | 11323.9 | **+30.28 tok/step** |
| `critic/score/mean` | −0.734 | −0.657 | −0.015 | +0.183 | +0.00249/step |
| `actor/entropy` | 0.873 | 0.742 | 0.142 | 0.118 | −0.00114/step |
| `rollout_corr/kl` | 0.0017 | 0.0015 | 0.0014 | 0.0020 | ~0 |
| `rollout_corr/rollout_is_eff_sample_size` | 0.9968 | 0.9972 | 0.9974 | 0.9966 | ~0 |
| `perf/time_per_step` | 202.9 s | 142.2 s | 196.6 s | 416.1 s | +0.43 s/step |

Cached series: `tmp_inspect_20260901/bf16_ref_0603.json` (regenerate any time from
the run ids above).

## What to compare against

Only compare at the **same global step** - these curves are strongly
step-dependent, and reward/length both move a lot between step 20 and step 265.

- **Response length growth is the primary signal.** BF16 gains +30 tok/step over
  the first 265 steps. The first verl real-W4A4 attempt lost −0.67 tok/step over
  420 steps, which is what exposed the wrong quantization scope and pre-packed
  refit. NeMo RL's all-MLP / no-3-loss arm gains +16.9 tok/step.
- `rollout_corr/kl` will *not* match. BF16 sits at ~0.0015 because train and
  rollout run the same precision; real W4A4 sits near 0.008 by construction.
  Watch its trend, not its absolute distance from BF16.
- Judge reward, accuracy, response length and train-rollout mismatch together.
  A low log-prob difference on its own is not evidence that an arm is better.
