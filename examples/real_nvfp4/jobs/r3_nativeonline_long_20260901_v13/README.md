# Real NVFP4 R3 long chain v13

Promotes the v12 runtime (validated v8 image + `enable_pdl=False`) into one
fresh, continuous 8-node long run. Contains no Slime three-loss implementation.

Four jobs target global steps 80, 150, 210 and 260 (shrinking chunks, because
step time tracks response length and every partition caps at 5h). They share one W&B run ID
and one checkpoint namespace, and are chained with `afterok`.

## Why this replaces v9

v9 carried the same v8 image and hit the intermittent startup hang twice. v12
removes that hang by disabling PDL on the two FlashInfer TRT-LLM NVFP4 MoE call
sites, without touching CUDA graphs, `max_num_seqs`, the quantization scope or
the reload path.

## Gates

Beyond v9's structural gates this bundle requires a recorded **numeric** gate.
v11 passed every structural gate while its rollout produced near-uniform
output, so `long_validate_static` now refuses to release until
`run_state/<version>/short_metrics.pass` exists:

```bash
python3 examples/real_nvfp4/jobs/r3_nativeonline_long_20260901_v13/audit_short_metrics.py \
  --run "$RN4PT_SHORT_EXP"_j<short_job_id> \
  --out  "$LONG_STATE"/short_metrics.pass
```

The audit refuses unless, over >= 20 logged steps: `rollout_corr/kl` <= 0.02,
ESS >= 0.97, `actor/entropy` <= 1.5, `response_length/clip_ratio` <= 0.10,
`actor/grad_norm` > 0 on every step, and the response-length slope is not
sharply negative. Reference band: v8 job 2694027.

```bash
bash examples/real_nvfp4/jobs/r3_nativeonline_long_20260901_v13/submit.sh audit
bash examples/real_nvfp4/jobs/r3_nativeonline_long_20260901_v13/submit.sh release
```

Compare the resulting curve against the BF16 reference run
`gb200_30B_bf16_megatron_0603` in `shawnzzz/DAPO-NVFP4-QAT`.
