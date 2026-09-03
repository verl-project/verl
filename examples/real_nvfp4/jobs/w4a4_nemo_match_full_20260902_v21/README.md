# W4A4 fully matched to NeMo RL's DAPO recipe

Three knobs, because the earlier arms proved they are not separable:
**strict Minerva scoring + DAPO dynamic sampling + no importance-sampling correction.**

## The arm matrix that got here

| arm | verifier | dyn. sampling | TIS | reward | length |
| --- | --- | --- | --- | --- | --- |
| v13 | lenient | off | on | −0.637 | flat ~1000 |
| v18 | lenient | off | **off** | −0.6 band | flat ~1090 @28 |
| v19 | **strict** | off | on | −0.914 | collapsed →488 @23 |
| v20 | **strict** | **on** | on | −0.931…−0.884 | collapsed →579 @7 |
| **v21** | **strict** | **on** | **off** | — | — |
| NeMo | strict | on | **none** | −0.875…−0.918 | **stable ~900**, takes off ~step 80 |

v20 reproduced NeMo's reward exactly yet its length fell 973 → 579 in seven steps
while NeMo holds ~900 at the same reward. The remaining difference was TIS:
NeMo's `loss_fn` carries no importance-sampling correction at all, and v20 still
ran verl's default `rollout_is=token`.

TIS is harmless under lenient scoring (v13 ≈ v18, both flat). But both strict
arms that collapsed had it on. Strict scoring makes correct samples rare, so the
gradient concentrates on few tokens, where reweighting by a quantization-derived
mismatch can do real damage.

## What this arm settles

- **Length stabilises then grows** → the plateau was a three-way recipe mismatch
  against NeMo, not W4A4. Pin all three in the formal recipe, re-run the long
  chain, and treat v13's 260 steps as not comparable to slime/NeMo.
- **Still collapses** → recipe alone cannot explain it; return to the W4A4
  numerics line, which requires porting verl's MoE weight sync to vLLM 0.26's
  RoutedExperts API so the BF16 control can run.

Early gates: step-1 `critic/score/mean` ≈ −0.9 (strict active), contract line
must read `tis=null`, and length must not repeat the v19/v20 collapse.

```bash
./submit.sh probe && ./submit.sh build && ./submit.sh preflight && ./submit.sh control
```
