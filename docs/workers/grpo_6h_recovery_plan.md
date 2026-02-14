# GRPO Recovery Plan (6 Hours)

## Objective

Within 6 hours, make DeepSpeed+GRPO path reproducible, non-stalling in practice, and PR-ready in code quality.

## Success Criteria

- A `60-step` GRPO run can start and produce stable `step` logs.
- `critic/score/mean` is non-zero on GSM8K settings intended for reasoning reward.
- No hidden worker crash (`Traceback`, `ActorDiedError`, `NCCL timeout`) in Ray logs.
- Updated docs include supported features and runnable commands.

## Timeline

### Phase 1 (0h-1h): Baseline and Instrumentation

- Reproduce with a control GRPO config and a candidate fix config.
- Add/enable lightweight diagnostics:
  - step-level timing (`timing_s/gen`, `timing_s/step`)
  - score/reward curves
  - memory reserved/allocated
- Confirm whether issue is true deadlock or oversized first-step workload.

Exit gate:

- Clear classification of failure mode with logs.

### Phase 2 (1h-3h): Code Fix and Config Stabilization

- Patch reward wiring so default reward path honors `reward.reward_kwargs`.
- Validate with compile check and smoke training.
- Tune GRPO config for stability:
  - response length and batch/token budget balance
  - rollout concurrency bounds

Exit gate:

- GRPO produces first steps with non-zero score on fix config.

### Phase 3 (3h-5h): 60-step Run and Benchmark Collection

- Launch 60-step GRPO run.
- Monitor progress and collect curves:
  - memory
  - throughput
  - score
  - loss
- If no step progress over threshold window, trigger diagnosis branch and fallback config.

Exit gate:

- 60-step run either completed, or has clear progress trend and diagnosis trail if still running.

### Phase 4 (5h-6h): PR Cleanup

- Tighten comments and remove dead/legacy branches in touched path only.
- Update docs:
  - feature matrix
  - six-case commands
  - GRPO command and caveats
- Summarize residual risks and next actions.

Exit gate:

- Patch is reviewable as PR with reproducible commands and artifacts.

## Diagnostics Checklist

- `outputs/<exp>/train.log`: `step`, `score`, `timing_s/gen`, `perf/throughput`.
- `/tmp/r_*/session_latest/logs`: worker-level hidden exceptions.
- `nvidia-smi`: active GPU utilization vs fake-stall.

## Risk Controls

- Keep control run and fix run side-by-side to avoid false positives.
- Do not claim deadlock before checking worker stacks/logs and GPU activity.
- Treat oversized first step as configuration/perf issue, not immediate correctness bug.
