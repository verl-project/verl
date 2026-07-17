# Profiling Examples

End-to-end GRPO runs that enable one of verl's profilers so you can capture a performance/memory trace without authoring a bespoke launcher. All scripts use the current `verl.trainer.main_ppo` entry point and the current Hydra API.

## Canonical Scripts

| Script                                  | Profiler          | Model              | Infer  | Train | Platform |
|-----------------------------------------|-------------------|--------------------|--------|-------|----------|
| `run_qwen3_8b_npu_profile_e2e.sh`       | NPU (E2E)         | Qwen3-8B           | vLLM   | FSDP  | NPU      |
| `run_qwen3_8b_npu_profile_discrete.sh`  | NPU (discrete)    | Qwen3-8B           | vLLM   | FSDP  | NPU      |
| `run_qwen2_5_vl_7b_torch_memory.sh`     | torch_memory      | Qwen2.5-VL-7B      | SGLang | FSDP  | NVIDIA   |
| `run_qwen2_5_7b_torch_profile.sh`       | torch (scheduled) | Qwen2.5-7B         | vLLM   | FSDP  | NVIDIA   |

### Torch profiling

- `run_qwen2_5_7b_torch_profile.sh` captures PyTorch profiler chrome traces (`.json.gz`) of the actor update loop. It demonstrates `torch.profiler.schedule`: the profiler advances one step per mini-batch and only records a `wait`/`warmup`/`active` window, repeated `repeat` times, rather than tracing every mini-batch.

Controlled via `global_profiler.tool=torch`, `global_profiler.steps=[...]`, `global_profiler.save_path=...`, plus per-role `actor_rollout_ref.actor.profiler.tool_config.torch.*` overrides. Override `PROFILE_STEPS`, `PROFILE_SAVE_PATH`, `PROFILE_RANKS`, `PROFILE_CONTENTS`, `PROFILE_DISCRETE`, and the schedule knobs `PROFILE_SCHEDULE_{SKIP_FIRST,WAIT,WARMUP,ACTIVE,REPEAT}` to adjust behavior. Set `PROFILE_SCHEDULE_ACTIVE=0` to disable scheduling and collect the whole window continuously. Load traces in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/). See [docs/perf/torch_profiling.md](../../docs/perf/torch_profiling.md) for details.

### NPU profiling

- `*_profile_e2e.sh` — one end-to-end timeline for all ranks.
- `*_profile_discrete.sh` — per-stage (rollout/ref/actor) discrete traces.

Controlled via `global_profiler.tool=npu`, `global_profiler.steps=[...]`, `global_profiler.save_path=...`, plus per-role `actor_rollout_ref.*.profiler.*` overrides. Override any of `PROFILE_STEPS`, `PROFILE_SAVE_PATH`, `PROFILE_LEVEL`, `PROFILE_CONTENTS`, `PROFILE_DISCRETE`, `PROFILE_RANKS_ALL` to adjust behavior.

### Torch memory profiling

- `run_qwen2_5_vl_7b_torch_memory.sh` dumps `torch.cuda._record_memory_history` snapshots to `global_profiler.save_path` (default `./mem_snapshots`). Load the `.pickle` in PyTorch's memory viz UI. Override `TRACE_ALLOC_MAX_ENTRIES`, `STACK_DEPTH`, `PROFILE_SAVE_PATH` as needed.

## Conventions

- `VAR=${VAR:-default}` for `MODEL_PATH`, batch sizes, learning rate, rollout TP, profile options, etc.
- Dynamic batch size and `trainer.balance_batch=True` are enabled by default.
- No deprecated config (`ppo_megatron_trainer.yaml`, `ppo_micro_batch_size`, `data.val_batch_size`, top-level `reward_model.*`, `actor.ulysses_sequence_parallel_size`).
