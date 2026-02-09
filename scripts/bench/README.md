# DeepSpeed Benchmark Scripts

This folder contains benchmark scripts used for ZeRO-stage regression checks and PR reporting.

## Scripts

- `scripts/bench/run_ds_zero_six_case_bench.sh`
  - Runs six cases: ZeRO1/2/3 with `offload=none/cpu`.
  - Default is `60` steps and one seed.
  - Output: `summary.tsv` with score, throughput, and GPU memory metrics.

- `scripts/bench/run_ds_zero12_seed_gate_bench.sh`
  - Runs ZeRO1 vs ZeRO2 crossover across seeds.
  - Produces paired comparisons and gate-style summary.
  - Output: `summary.tsv`, `compare.tsv`, `overall.tsv`.

- `scripts/bench/build_ds_zero2_technical_report.py`
  - Builds a markdown technical report from benchmark output folders.
  - Supports six-case runs and stage1/2 gate runs.

## Recommended Usage

```bash
# 1) six-case benchmark (seed 7777, 60 steps)
SEED=7777 STEPS=60 RUN_TAG=zero_six_60_seed7777_fix \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 ZERO2_STEP_EACH_MICRO=0 \
bash scripts/bench/run_ds_zero_six_case_bench.sh

# 2) stage1/2 crossover (step_each_micro=1)
SEEDS="7777" STEPS=60 RUN_TAG=ds_stage12_seed60_gate_stepmicro1 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 VERL_DS_ZERO2_STEP_EACH_MICRO=1 \
bash scripts/bench/run_ds_zero12_seed_gate_bench.sh

# 3) build report
python3 scripts/bench/build_ds_zero2_technical_report.py \
  --out-dir outputs/zero2_pr_report \
  --six-run outputs/<six_run_dir> \
  --stage12-run outputs/<stage12_run_dir>
```

## Notes

- Benchmark outputs are not committed; keep them under `outputs/`.
- If a run is interrupted, inspect each case `train.log` first; reruns are retry-safe.
