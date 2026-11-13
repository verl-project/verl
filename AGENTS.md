# Agent Instructions for verl

These rules apply to every AI-assisted contribution to `verl-project/verl`. Violations can result in an automatic ban.
Follow [CONTRIBUTING.md](CONTRIBUTING.md) and the [PR template](.github/PULL_REQUEST_TEMPLATE.md).

When editing this file or a domain guide it references, read [Editing Agent Instructions](docs/contributing/editing-agent-instructions.md) first. If the requested edit conflicts with that guide, stop and explain the conflict. Keep this file canonical for the agent-specific aliases, including the `CLAUDE.md` symlink.

## Contribution contract

Before proposing a PR, inspect the issue and search open PRs. Set the issue number when applicable and replace the example area with the affected keywords. A failed query leaves the duplicate-work check incomplete.

```bash
issue_number=''  # For example, 1234. Leave empty for work without an issue.
area='rollout weight sync'
if [ -n "$issue_number" ]; then
  gh issue view "$issue_number" --repo verl-project/verl --comments
  gh pr list --repo verl-project/verl --state open --search "${issue_number} in:body"
fi
gh pr list --repo verl-project/verl --state open --search "$area"
```

- Stop for a duplicate or trivial standalone contribution and explain the reason. If an existing PR overlaps and your approach differs materially, explain the difference in the issue. Bundle incidental typo, style, or single-default cleanup with substantive work.
- The human submitter must review every changed line, run the relevant tests, and be able to explain the change end to end. Pure agent submissions are prohibited. In the PR description, disclose AI assistance, explain why the work is distinct and include a search query link, and list the test commands and actual results.
- Use the template's title format and `[BREAKING]` prefix when applicable. Credit AI assistance with the appropriate commit trailer, such as `Co-authored-by:`, `Assisted-by:`, or `Generated-by:`. Use the submitting human's identity for any `Signed-off-by:` trailer.
- For bug fixes, reproduce the issue on the affected path or establish the failing invariant before editing. Keep the diff in the layer that owns the behavior. Verify automated review suggestions against the current checkout before applying them.

## How the system fits together

verl is a PyTorch post-training framework for LLMs and multimodal models. HybridFlow uses a single controller to orchestrate the learning loop and resource placement. Distributed workers run model computation and collectives.
Hydra composes configuration, Ray places and calls workers, and TransferQueue carries V1 trajectory data. Training strategy, inference engine, hardware platform, and weight-sync transport are separate choices. Use the launcher and matching CI job to identify which combinations are exercised.

| Boundary | What to search for |
| --- | --- |
| Launch and configuration | Find `run_ppo` and `class TaskRunnerV1`. Follow the `ppo_trainer` Hydra config through the `model_engine` group and the launcher's call to `validate_config`. |
| Trainer and algorithm | Find `class PPOTrainer` and its `_step_once` method. Search `register_adv_est` and `register_policy_loss` for algorithm extensions, and `compute_rollout_correction_and_add_to_batch` for importance sampling and rejection. |
| Placement and RPC | Find `ResourcePoolManager`, `RayWorkerGroup`, and `create_colocated_worker_cls`. Trace mesh-specific dispatch and collection through `make_nd_compute_dataproto_dispatch_fn`. |
| Distributed training and model code | Find `TrainingWorker` and `ActorRolloutRefWorker`. Follow `EngineRegistry` to the selected implementation of `forward_backward_batch` and `get_per_tensor_param`. |
| Requests and inference | Find `LLMServerManager` and `get_router_handle` for routing. Search `get_rollout_replica_class` for replica selection, `TokenOutput` for generation results, and `get_rollout_class` for weight-loading adapters. |
| Prompt sources and trajectories | Find `RLHFDataset` and `PromptDataLoader` for inputs. Follow `AgentLoopWorkerTQ` and `ReplayBuffer` through trajectory storage and sampling, and `tqbridge` for worker-side materialization. Search `class DataProto` for the legacy batch representation. |
| Agent loop, reward, and teacher | Find `AgentLoopBase` for rollout extensions, `RewardLoopManager` and `get_custom_reward_fn` for scoring, and `MultiTeacherModelManager` and `register_distillation_loss` for distillation. |
| Weights and checkpoints | Find `CheckpointEngineManager` and `WeightUpdateTicket` for live rollout sync, `BaseCheckpointManager` for persistent save and resume, and `BaseModelMerger` for Hugging Face export. |
| Other entry points | Search `sft_trainer_engine` for the `torchrun` and Ray SFT entry points, `generate_per_replica` for offline generation, and `test_score/` for offline evaluation. See [Examples](examples/README.md) for supported launchers. |

For design background, read [HybridFlow](docs/hybrid_flow.rst), [single-controller dispatch](docs/single_controller.rst), [engine workers](docs/workers/engine_workers.rst), and the [Ray tutorial](examples/tutorial/ray/tutorial.ipynb). The HybridFlow walkthrough uses the legacy `RayPPOTrainer` path. Use the trace below for the V1 entry point.

## Trace the active training path

1. **Choose the trainer.** `python -m verl.trainer.main_ppo` composes `ppo_trainer`, selects the device, validates config and starts or reuses Ray. `trainer.use_v1=true` selects V1 and `trainer.v1.trainer_mode=sync` is the default. V1 initializes TransferQueue and selects the trainer from its registry. With `trainer.use_v1=false`, follow the legacy task runner into `RayPPOTrainer.init_workers` and `RayPPOTrainer.fit`. Older examples, recipes, and experimental async trainers can take that path. Resolve the actual launcher's config before deciding which code to edit.
2. **Build roles and place workers.** The trainer places actor, rollout and optional critic, reference, reward model, and teacher. See the `need_*` helpers in `verl/trainer/ppo/utils.py`. GAE enables the critic by default unless explicitly overridden. KL in the reward or actor loss needs a reference policy, which can be served by the actor with adapters disabled in the LoRA path. Worker groups bind RPC dispatch to role and mesh. The training driver and accelerator workers can have different device visibility.
3. **Generate and store trajectories.** The dataset supplies chat messages, metadata and optional multimodal inputs. The agent loop applies the chat template and calls the rollout client. `AgentLoopWorkerTQ` fans each prompt into sessions and writes trajectories to `train` or `val` with keys of the form `{uid}_{session_id}_{index}`. The prompt record tracks group completion. Reward may be computed in the agent loop or after sampling for a colocated reward model. Optional teacher log probabilities support distillation. `ReplayBuffer` and `ReplayBufferAsync` select groups and handle refill, filtering and staleness under the applicable mode.
4. **Execute a learning update.** In V1, `_step_once` samples metadata, fills colocated rewards if needed, pads and balances data-parallel work, obtains old log probabilities and optional reference log probabilities and critic values, then computes advantages and returns. It updates the critic when enabled and updates the actor once `global_steps >= critic_warmup`. Workers materialize selected fields through `tqbridge` and operate on `TensorDict`. Follow `compute_log_prob` / `update_actor` → `TrainingWorker.infer_batch` / `train_mini_batch` → engine and `verl/workers/utils/losses.py`. The V0 driver uses `DataProto` and converts batches at the shared worker boundary. Trace both callers when changing that boundary.
5. **Publish weights and advance the lifecycle.** In `sync` and `colocate_async`, training and rollout share devices. The mode hooks control when rollout sleeps, receives weights, and resumes. `colocate_async` supports partial generation across updates. `separate_async` requires a standalone pool and a non-`naive` transport. Hybrid replicas are opt-in via `trainer.v1.separate_async.hybrid_rollout.enable_switch` and require `actor_rollout_ref.hybrid_engine=true`. These replicas use the colocated `naive` path. Step switching and rollout prefill/decode disaggregation cannot be combined. See the [V1 trainer guide](docs/advance/v1_async_trainer.md).

## Invariants to enforce when editing

- **Config and public interfaces.** Trace the Hydra defaults, override keys, `_target_` dataclass, validation, and actual consumer. For PPO engine changes, compose the top-level `model_engine` group and check the resulting actor, critic and reference strategies. Keep defaults compatible with existing launchers and checkpoints unless deliberately changing the contract. For trainer YAML covered by the config-format check, use comments above keys with blank lines between fields. Edit source YAML and regenerate the `_generated_ppo_*.yaml` snapshots.
- **Tensor and metadata contracts.** Preserve sample identity and field alignment across padding, packing, balancing and collection. Apply any reorder consistently to tensors, metadata and queue keys or tags. Preserve order where a caller relies on it. Document shapes, masks, dtypes and optional fields at changed boundaries. Use the padding helpers in `verl/workers/utils/padding.py`. Keep `DataProto` out of model engines. Preserve the distinction between tokens visible to the model and tokens eligible for loss. Keep GRPO groups keyed by prompt UID through reorder, filtering and refill.
- **RL numerics.** Treat rollout behavior log probabilities, the old-policy anchor, current actor log probabilities and the KL reference as distinct fields. Only configured bypass mode substitutes rollout log probabilities for the old-policy anchor. Preserve temperature, response masks and reduction denominators across workers and microbatches. Exercise a deterministic small batch with unequal response lengths and verify masked positions and loss/gradient scaling.
- **Distributed execution.** Use the existing worker RPC and mesh helpers. Check which ranks execute, which collect, and whether a call blocks. Preserve process-group membership, collective order across ranks, and divisibility after data-parallel sharding. Check rank-local batch or token counts before changing aggregation. Investigate the first failing rank before modifying timeouts or adding barriers.
- **Generation and weight sync.** Follow `RolloutReplica` for placement/lifecycle, the backend server for sampling and `TokenOutput`, and the backend `ServerAdapter` for weight loading. Verify response budgets, terminal reasons, and token/log-probability alignment across retries. Trace live parameters from the training engine's `get_per_tensor_param` through transport and adapter. Preserve the wire format, version tags, cache invalidation and abort/sleep/update/resume ordering. A published training snapshot can precede rollout commit. Use `wait_for_weight_update` where the caller needs the committed rollout version. Verify normal update, repeat update, and failure or resume paths when changing this contract.
- **Persistent resume.** Trace `_save_checkpoint`, `_load_checkpoint`, and `_reissue_inflight_prompts` for V1 changes. Verify that model and optimizer state, data position, and any enabled TransferQueue snapshot agree on the resumed step. Exercise pending or running prompt recovery if the change affects async state.
- **Extensions and dependencies.** Register algorithms, engines and adapters at their owning boundary. Keep backend-specific imports and patches scoped to that backend, and preserve CPU import and config composition. Use the [dependency upgrade workflow](.agent/skills/upgrade-deps/SKILL.md) when changing locked dependencies. Regenerate `uv.lock` from `pyproject.toml` and review paired image or ABI changes.
- **Code style.** Use the repository's Ruff, Ruff format, and mypy settings in `pyproject.toml` (120-column Python). Match nearby docstrings, types and license headers. Use `verl` and `SGLang` in prose, `sglang` in config. Follow [example naming and override conventions](examples/README.md) and the device checks in [pre-commit](.pre-commit-config.yaml).

For an async stall, trace prompt status in the correct TransferQueue partition, then check whether the replay buffer can sample the group. If generation stopped during an update, inspect the preceding weight commit and abort/resume state. For batching errors, write down the units of each setting (prompts, trajectories, tokens, or per-rank samples). Check where `rollout.n` expands the prompt batch and where the trainer pads or balances it.

## Hardware platforms and backend selection

Read the registrations in `verl/plugin/platform/platform_manager.py` and `EngineRegistry.get_engine_cls` for current behavior. `VERL_PLATFORM` overrides platform probing. Engine lookup takes model type and training backend, then tries an exact device/vendor key, a device default, and the NVIDIA CUDA registration for a compatible CUDA device. `VERL_ENGINE_DEVICE` and `VERL_ENGINE_VENDOR` override engine selection. The NVIDIA fallback on a CPU-only actor does not establish CUDA availability.

| In-tree platform | Integration and where to validate |
| --- | --- |
| NVIDIA GPU (`nvidia`, CUDA) | `PlatformCUDA` supplies the Ray `GPU` resource, device visibility and the default NCCL process-group backend. Match training (FSDP/FSDP2, Megatron, VeOmni, TorchTitan, or AutoModel where registered) with the tested rollout stack. vLLM, SGLang and TensorRT-LLM have async replica registrations. TensorRT-LLM uses its own image. See [installation](docs/start/install.rst) and [CUDA CI jobs](.github/workflows/). |
| Huawei Ascend (`huawei`, NPU) | `PlatformNPU` uses `torch_npu`, Ray `NPU`, Ascend device visibility and the default HCCL process-group backend. Megatron on NPU routes through MindSpeed, whose patches must load before Megatron initialization. vLLM uses vLLM Ascend, and SGLang has a separate Ascend stack. The checkpoint-engine key `nccl` selects the HCCL implementation on Ascend. Check the CANN/chip matrix and transport support. See the [Ascend backend guide](docs/ascend_tutorial/en/feature_support/ascend_backend_features.md). |
| AMD Instinct (`amd`, ROCm/HIP) | `PlatformROCm` extends `PlatformCUDA` and identifies HIP builds via `torch.version.hip`. PyTorch device type remains `cuda`. Respect Ray's HIP/ROCR visibility handling and use the image's RCCL and kernel stack. The [ROCm PPO workflow](.github/workflows/e2e_ppo_trainer_megatron_vllm_rocm.yml) exercises FSDP/FSDP2 and Megatron with vLLM. Check the [AMD quick start](docs/amd_tutorial/amd_quick_start.rst) and `docker/rocm/` before selecting a ROCm image or testing SGLang. |

Use `verl.utils.device` or `get_platform()` in shared code for device, memory, visibility and collective APIs. For vendor extensions use `PlatformRegistry` and `EngineRegistry` via `verl.plugins` entry points or `VERL_USE_EXTERNAL_MODULES`. Intel XPU, Cambricon MLU, MetaX and Google TPU use external/reference integrations. See [multi-chip support](docs/hardware/multi_chip_support.rst) for the extension model and the live registry for the current built-ins. A CPU test environment provides config and logic coverage.

## Environment and verification

Use `uv` for Python environment management. If it is missing, use the [official installer](https://docs.astral.sh/uv/getting-started/installation/).
The locked CPU and CUDA workflows target Linux with Python 3.12. Select extras using the conflict rules in `pyproject.toml` and keep the selection consistent for the driver and Ray workers. Use at most one inference extra in one environment. For Ascend, ROCm and TensorRT-LLM, use the matching image and its interpreter. Set `VERL_USE_UV=0` when a shared launcher would otherwise select the CUDA lock on one of those images.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install pre-commit hydra-core
pre-commit install
pre-commit run --all-files --show-diff-on-failure
```

```bash
# Locked CPU regression example. Substitute tests that cover your change.
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
  uv run --frozen --all-packages --extra cpu pytest -q --asyncio-mode=auto \
  tests/trainer/ppo/test_core_algos_on_cpu.py tests/test_protocol_on_cpu.py
# Print the composed PPO config with a rollout backend selected.
uv run --frozen --all-packages --extra cpu python -m verl.trainer.main_ppo \
  --cfg job --resolve model_engine=dp actor_rollout_ref.rollout.name=vllm
# Regenerate and check flattened configs in the active environment.
pre-commit run autogen-trainer-cfg --all-files
```

Hydra's `--cfg job --resolve` prints the config and exits before entry-point validation, Ray startup or engine initialization. Check required fields for enabled components and run the matching entry point when testing runtime behavior.

For a locked CUDA checkout with FSDP and vLLM, the environment command is `uv sync --frozen --all-packages --extra fsdp --extra vllm`.
Use matching extras when passing a Python command through `uv run`. Canonical PPO and GRPO launchers live under `examples/ppo_trainer/` and `examples/grpo_trainer/`. Read the launcher's inputs and matching CI job before allocating devices or using default dataset paths.

| Change | Evidence to collect |
| --- | --- |
| Algorithm, loss, config or batch schema | Run focused tests under `tests/trainer/ppo/`, `tests/workers/config/` or `tests/test_protocol*_on_cpu.py`. For config edits, compose the actual launcher overrides and run the config generator. The generator rewrites snapshots and returns nonzero when they differ from the Git index. Review the regenerated output and include it with the source edits. |
| V1 queue, async or resume | Start with `tests/trainer/ppo/v1/` and `tests/utils/test_transferqueue_utils_on_cpu.py`. For placement, async progress or weight-version changes, use the relevant [V1 workflow](.github/workflows/e2e_v1_separate_async.yml) and `tests/special_e2e/run_v1_separate_async.sh` or its platform-specific counterpart. |
| Engine, model, serving, or transport | Use the relevant [GPU](.github/workflows/gpu_unit_tests.yml), [NPU](.github/workflows/npu_unit_tests.yml), [vLLM](.github/workflows/vllm.yml), [SGLang](.github/workflows/sgl.yml), or [TensorRT-LLM](.github/workflows/e2e_ppo_grpo_trainer_trtllm.yml) job for setup, data, device count and exclusions. For transport, start in `tests/checkpoint_engine/`. Test real devices and ranks for changes to collectives, placement, kernels or weight transfer. |
| Docs or examples | Run the relevant hooks. Use `make -C docs html` with the [docs dependencies](docs/requirements-docs.txt) for Sphinx changes. For launchers run `bash -n` and compose the actual overrides, then smoke the workload if runtime behavior changed. |

Mirror package layout under `tests/`. The [CPU workflow](.github/workflows/cpu_unit_tests.yml) discovers files ending `_on_cpu.py`. Some of these tests still need local datasets or a working local Ray/TransferQueue runtime.
Inspect workflow path filters when adding coverage. Run the smallest relevant tests first and choose end-to-end evidence that exercises the changed path.
Check `git diff --check` and review hook edits. Report commands, results, skips and blockers. For accelerator runs record the device, engine versions and rank topology. For numeric or performance changes, report a baseline and candidate on the same inputs, with tolerances or steady-state measurements as appropriate.

## Offline dataset inference

Start with [the generation example](examples/generation/run_deepseek_llm_7b.sh) and [GSM8K preprocessing](examples/data_preprocess/gsm8k.py). Use a nonempty Parquet file with a `prompt` column of chat-message lists. Keep the number of messages per row uniform for this launcher. Its NumPy conversion rejects ragged chat lists. `main_generation_server.start_server` creates standalone rollout replicas, and `generate` distributes requests to their `/v1/chat/completions` endpoints. The entry point writes one row per input in input order, with `rollout.n` responses per row. Use a vLLM environment that includes the OpenAI Python client.

```bash
# Run inside the prepared vLLM environment with local model and dataset files.
HF_HUB_OFFLINE=1 MODEL_PATH=/path/to/local/model \
DATA_PATH=/path/to/prompts.parquet OUTPUT_PATH=/path/to/responses.parquet \
NNODES=1 NGPUS_PER_NODE=1 ROLLOUT_TP=1 N_SAMPLES=1 \
  bash examples/generation/run_deepseek_llm_7b.sh \
  actor_rollout_ref.model.trust_remote_code=False
```

Choose a local chat model that fits the allocation and can load with `trust_remote_code=False`. Provision model, tokenizer, data and dependencies before disconnecting. Make the model and tokenizer paths available on each rollout node, and make the input and output paths accessible to the driver. Set `HF_HUB_OFFLINE=1` in the worker environment when using an existing cluster. The example sets `load_format=auto` to load real weights. The training default `dummy` assumes a subsequent weight sync.

The launcher sizes replicas as total devices divided by tensor parallel size (TP). Keep rollout data and pipeline parallel sizes at one, make TP divide the allocation, and ensure each cross-node replica uses an integer number of full node allocations. With temperature zero use one sample. The CLI forwards temperature, top-p, and response length as `max_tokens`. It currently omits top-k from the HTTP request.

Append `--cfg job` for a composition-only check. For a runtime smoke, use a few local rows and verify row order, response-list lengths, and a nonempty completion for a normal prompt. To score existing completions, inspect `evaluation.yaml` and use `python -m verl.trainer.main_eval` with the Parquet path and a custom reward function. The evaluator also requires the source and ground-truth columns.

## Acknowledgements

The original contribution policy was adapted from [vLLM's Agent Instructions](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).
