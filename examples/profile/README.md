# Profiling Examples

End-to-end GRPO runs that enable one of verl's profilers so you can capture a performance/memory trace without authoring a bespoke launcher. All scripts use the current `verl.trainer.main_ppo` entry point and the current Hydra API.

## Canonical Scripts

| Script                                  | Profiler          | Model              | Infer  | Train | Platform |
|-----------------------------------------|-------------------|--------------------|--------|-------|----------|
| `run_qwen3_8b_npu_profile_e2e.sh`       | NPU (E2E)         | Qwen3-8B           | vLLM   | FSDP  | NPU      |
| `run_qwen3_8b_npu_profile_discrete.sh`  | NPU (discrete)    | Qwen3-8B           | vLLM   | FSDP  | NPU      |
| `run_qwen2_5_vl_7b_torch_memory.sh`     | torch_memory      | Qwen2.5-VL-7B      | SGLang | FSDP  | NVIDIA   |
| `run_qwen2_5_7b_torch_profile.sh`       | torch             | Qwen2.5-7B         | vLLM   | FSDP  | NVIDIA   |

### Torch profiling

- `run_qwen2_5_7b_torch_profile.sh` captures PyTorch profiler chrome traces (`.json.gz`) of **both** the actor update loop (training) and the vLLM rollout engine (inference):
  - **Training** is collected continuously, so one file per profiled step per rank holds the whole step as that worker ran it, with `compute_log_prob`, `compute_ref_log_prob` and `update_actor` as named rows inside it, the update loop's `mini_batch<i>` rows nested under `update_actor`, and each stage's engine micro-batches shown as `micro_batch<j>` rows (a forward-only log-prob stage that fits in one micro-batch shows a single `micro_batch0`). Traces land directly under `save_path/` and carry no `ProfilerStep#<n>` rows. To keep the file small when the update loop has many mini-batches, set `PROFILE_SCHED_ACTIVE=N` to record only the first `N` update mini-batches (every other stage stays in full); `discrete=True` gives the update stage its own trace where the full `skip_first`/`wait`/`warmup`/`active`/`repeat` schedule applies.
  - **Inference (rollout)** is collected by vLLM's own engine-side torch profiler, which only runs in *discrete* mode. The script therefore forces `rollout...torch.discrete=True` (independent of the actor) and captures the full `generate_sequences` window on each profiled step. The engine writes into `save_path/agent_loop_rollout_replica_<rank>/`, and `global_profiler.relocate_results` (on by default here, `PROFILE_RELOCATE=False` to opt out) moves those traces up into `save_path/` as `rollout-replica<rank>_...` once the step finishes.

Controlled via `global_profiler.tool=torch`, `global_profiler.steps=[...]`, `global_profiler.save_path=...`, plus per-role `actor_rollout_ref.{actor,rollout}.profiler.tool_config.torch.*` overrides. Override `PROFILE_STEPS`, `PROFILE_SAVE_PATH`, `PROFILE_RANKS`, `PROFILE_CONTENTS`, `PROFILE_DISCRETE`, `PROFILE_SCHED_ACTIVE` (and `PROFILE_SCHED_{SKIP_FIRST,WAIT,WARMUP,REPEAT}` for discrete mode) to adjust training behavior. For inference, set `PROFILE_ROLLOUT=False` to profile training only, or `PROFILE_ROLLOUT_TOKEN_{START,END}` to restrict rollout tracing to a response-token window. `PROFILE_FINISH_HOOK_CMD` runs a command **once, after the last profiled step**, on each selected rank, e.g. to upload the traces off the node. Backend stop and `relocate_results` still run every profiled step, so all steps' traces have accumulated in `save_path` by then; because the command runs a single time (not once per step), a command that uploads the whole directory (`'my-upload-tool "$VERL_PROFILE_SAVE_PATH"'`) sends each trace exactly once. `save_path` is usually node-local, so the command runs on every selected rank/node — set `PROFILE_RANKS` to one rank per node so each node's directory is uploaded once. Load traces in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/). See [docs/perf/torch_profiling.md](../../docs/perf/torch_profiling.md) for details.

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
