# BF16 control on the v12 stack

Single-variable control for the flat response-length result of the W4A4 long run
(`verl_30b_realw4a4_r3_nativeonline_8n_long_20260901_v13`).

## Why this exists

Over 210 steps the W4A4 arm's response length stayed flat (1015 -> 1124,
+0.31 tok/step) while three reference curves took off after step ~80:

| arm | steps 81-120 slope | steps 161-210 length |
| --- | --- | --- |
| verl BF16 `gb200_30B_bf16_megatron_0603` | +55.9 | 6158 |
| NeMo-RL W4A4 all-MLP 0of3 R3-on | +52.0 | 4432 |
| NeMo-RL W4A4 all-MLP 0of3 R3-off | +30.1 | 4027 |
| **verl W4A4 v13** | **+0.4** | **1037** |

NeMo-RL reaching 4k on the same precision contract rules out "FP4 cannot
lengthen". But `gb200_30B_bf16_megatron_0603` is **not** a clean control: it
differs from the W4A4 arm in precision *and* recipe (dynamic sampling on,
`gen_batch_size` 64 vs 32) *and* stack (June `verl_megatron_qat_v020`).

Dynamic sampling was checked first and largely ruled out: NeMo logs only
~4% degenerate prompt groups (`pct_0` 2.6-5.3%, `pct_1` 0.0%, `pct_mixed`
94.7-97.4%), so filtering them cannot account for a flat-versus-8x difference.

This bundle therefore changes exactly one thing against the W4A4 arm:
`PRECISION_MODE=bf16`. Same v12 image, same dataset adapter, same
`filter_groups.enable=False`, same `gen_batch_size=32`, same clip/lr/TIS/R3.

## Reading the result

- **Control grows** -> the gap is specific to verl's W4A4 path, and the next
  step is to diff verl's W4A4 against NeMo's W4A4 numerics.
- **Control also stays flat** -> precision is exonerated and the cause is in the
  shared verl recipe/plumbing; compare against the June BF16 run to find which
  recipe difference matters.

`PRECISION_MODE=bf16` sets `actor_rollout_ref.actor.megatron.real_nvfp4.enable=False`,
and the rollout mirrors it through
`rollout.real_nvfp4=${oc.select:actor_rollout_ref.actor.megatron.real_nvfp4,null}`,
so vLLM runs BF16 too. `train.job` asserts that neither attestation fires and
that `nvfp4_per_token` was not selected, because a "BF16" baseline that silently
keeps a W4A4 rollout has bitten this workspace before.

```bash
./submit.sh control          # 8 nodes, 100 steps (CONTROL_STEPS to override)
```
