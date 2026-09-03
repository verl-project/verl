# W4A4 with strict Minerva scoring

Single-variable arm against the v13 long run. Only the math verifier changes.

## The finding this tests

verl's `math_dapo.verify()` gained a `\boxed{}` fallback sometime after June:

```python
correct, pred = is_correct_minerva(solution_str, answer)   # needs "Answer: ..."
if pred != "[INVALID]": return correct, pred
box_correct, box_pred = is_correct_strict_box(...)          # <- added upstream
```

verl's own June tree (`verl_megatron_qat_v020`, which produced the BF16 0603
baseline) and NeMo RL's `dapo_math_verifier.py` both stop at the Minerva line.

The DAPO prompt demands `Answer: \boxed{$Answer}`. A base model emits bare
`\boxed{}`. Measured on run v13's logged generations:

| step | n | has `Answer:` | boxed-only (fallback credit) |
| --- | --- | --- | --- |
| 9 | 2 | 0 | 2 |
| 49 | 10 | 0 | 10 |
| 129 | 10 | 0 | 10 |
| 209 | 12 | 0 | 12 |
| 259 | 10 | 0 | 10 |
| **total** | **44** | **0 (0%)** | **44 (100%)** |

So **the entire reward signal came from the fallback**, and every one of those
responses scores wrong under the strict verifier.

## Why this explains the plateau

Under strict scoring the base model's native output earns nothing, so it must
first learn the required format - and producing well-formed complete solutions
co-occurs with longer structured reasoning. Under the fallback it is credited
immediately, starts near its ceiling, and has no pressure to change anything.

This accounts for every curve without invoking precision:

| arm | verifier | val acc | length |
| --- | --- | --- | --- |
| verl BF16 0603 (June tree) | strict | — | grows 759 -> 6606 |
| NeMo W4A4 | strict | 0.012 -> 0.508 | grows -> 4027 |
| **verl W4A4 v13 (current tree)** | **fallback** | **0.133, no trend** | **flat 1015 -> 1124** |

It also explains the 8x gap in starting accuracy (0.133 vs 0.012) and why NeMo's
accuracy climbs 40x while verl's oscillates.

Not W4A4, not TIS, not the vLLM upgrade - an upstream verl scoring change.

## Reading the result

- **Length takes off** -> confirmed; the recipe must pin strict scoring to
  reproduce the NeMo/slime baselines, and the earlier v13 numbers are not
  comparable to them.
- **Still flat** -> the verifier is not sufficient on its own; re-open the W4A4
  numerics line and unblock the BF16 control.

Watch the first 3 steps: `critic/score/mean` should start much **lower** than
v13's -0.637 (most responses now score wrong). If it does not, the env var did
not reach the reward workers - the scorer runs inside Ray, so it is passed via
`runtime_env_strict.yaml`, not a driver-side export.

```bash
./submit.sh probe && ./submit.sh build && ./submit.sh preflight && ./submit.sh control
```
