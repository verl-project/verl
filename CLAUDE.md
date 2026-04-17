# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is verl?

verl (Volcano Engine Reinforcement Learning) is a distributed RL training framework for LLMs built on Ray. It supports multiple training backends (FSDP/FSDP2, Megatron-LM), multiple rollout engines (vLLM, SGLang), and implements PPO, GRPO, REINFORCE++, RLOO, ReMax, DAPO, and other RL algorithms.

## Build & Install

```bash
# Recommended: use uv for environment management
uv venv --python 3.12
source .venv/bin/activate

# Editable install with test + inference engine
pip install -e .[test,vllm]    # or .[test,sglang]

# Other optional extras: [gpu], [mcore], [math], [prime], [geo], [trtllm]
```

## Linting & Formatting

```bash
# Setup pre-commit hooks (required before first commit)
pip install pre-commit hydra-core
pre-commit install

# Run all hooks
pre-commit run --all-files

# Run specific hooks
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

Ruff config: line-length=120, rules E/F/UP/B/I/G. mypy strict only on `verl.trainer.config.algorithm`, `verl.trainer.ppo.core_algos`, `verl.trainer.ppo.reward`, `verl.workers.reward_manager`.

The `autogen-trainer-cfg` hook regenerates `_generated_*.yaml` files from Python dataclasses. If you modify config dataclasses, run this hook before committing.

## Testing

```bash
# CPU unit tests (files named *_on_cpu.py)
echo '[pytest]\npython_files = *_on_cpu.py' > pytest.ini
pytest -s -x --asyncio-mode=auto tests/

# Run a single CPU test file
pytest -s -x --asyncio-mode=auto tests/test_protocol_on_cpu.py

# GPU unit tests (excludes CPU-only, special, and engine-specific tests)
pytest -s -x \
  --ignore-glob="*on_cpu.py" \
  --ignore-glob="*test_special_*.py" \
  --ignore-glob="tests/special*" \
  tests/

# Multi-GPU distributed tests (via torchrun)
torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/workers/actor/test_special_dp_actor.py

# E2E training tests are shell scripts in tests/special_e2e/
```

Test naming conventions:
- `*_on_cpu.py` — CPU-only tests (run in CPU CI)
- `test_special_*.py` — distributed tests requiring `torchrun`
- `tests/special_e2e/*.sh` — end-to-end training scripts

## Architecture Overview

### Core Data Type: `DataProto` (`verl/protocol.py`)
Universal data container passed between all workers. Wraps a `TensorDict` (`batch`) and `dict` of non-tensor data (`non_tensor_batch`), plus `meta_info`. Key fields: `prompts`, `responses`, `attention_mask`, `old_log_probs`, `ref_log_prob`, `values`, `advantages`, `returns`, `token_level_scores`, `response_mask`.

### Training Loop: `RayPPOTrainer` (`verl/trainer/ppo/ray_trainer.py`)
Central orchestrator running on the Ray driver (CPU). Manages `RayWorkerGroup` instances for each `Role` (Actor, Critic, Rollout, RefPolicy, RewardModel). The training loop per step: rollout → sleep replicas → compute log probs → compute ref log probs → compute values → compute advantages → update actor → update critic → checkpoint.

Entry point: `verl/trainer/main_ppo.py` — initializes Ray, builds workers, calls `RayPPOTrainer.fit()`.

### Role System (`verl/trainer/ppo/utils.py`)
`Role` enum: `Actor`, `Rollout`, `ActorRollout`, `Critic`, `RefPolicy`, `RewardModel`, `ActorRolloutRef`. `ActorRolloutRef` is the "hybrid engine" that co-locates actor training + rollout inference (+ optionally reference policy) on the same GPUs using sleep/wake memory sharing.

### Worker/Engine System
- **Single Controller** (`verl/single_controller/`): Ray-based RPC dispatch. `@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)` decorator marks worker methods. `Dispatch` enum controls data splitting: `RANK_ZERO`, `ONE_TO_ALL`, `ALL_TO_ALL`, `DP_COMPUTE`, `DP_COMPUTE_PROTO`.
- **New engine path** (`verl/workers/engine_workers.py`): `TrainingWorker` wraps any `BaseEngine`. This is the default path.
- **Legacy path** (`verl/workers/fsdp_workers.py`, `megatron_workers.py`): controlled by `trainer.use_legacy_worker_impl`.

### Training Backends (`verl/workers/engine/`)
`BaseEngine` (`base.py`) defines the interface. `EngineRegistry` maps `(model_type, backend)` to implementation.
- **FSDP/FSDP2**: `FSDPEngine` — default, recommended
- **Megatron-LM**: `MegatronEngine` — for large-scale tensor/pipeline parallelism
- **TorchTitan**, **VeOmni**, **Diffusers**: specialized backends

### Rollout Engines (`verl/workers/rollout/`)
`BaseRollout` (`base.py`) is the abstract base. Implementations: `vllm_rollout/`, `sglang_rollout/`, `trtllm_rollout/`, `hf_rollout.py`. Weight resharding between training (FSDP-sharded) and inference (TP-replicated) is managed by `ShardingManager` (`verl/workers/sharding_manager/`).

### Algorithm Implementations (`verl/trainer/ppo/core_algos.py`)
Pure functions (no distributed dependencies). `AdvantageEstimator` enum: `GAE`, `GRPO`, `REINFORCE_PLUS_PLUS`, `REMAX`, `RLOO`, etc. Extensible via `register_adv_est` / `register_policy_loss` decorators.

### Config System (Hydra/OmegaConf)
Primary config: `verl/trainer/config/ppo_trainer.yaml`. Composable sub-configs in `verl/trainer/config/{actor,critic,rollout,model,data,reward,algorithm,engine}/`. Typed dataclasses inherit from `BaseConfig` (`verl/base_config.py`), a frozen dataclass with dict-like interface. Key top-level sections: `actor_rollout_ref`, `critic`, `algorithm`, `trainer`, `reward`, `data`.

### Reward System (`verl/workers/reward_manager/`)
`AbstractRewardManager` base class. Implementations: `NaiveRewardManager`, `BatchRewardManager`, `PrimeRewardManager`, `DAPORewardManager`.

### Experimental Features (`verl/experimental/`)
`agent_loop/` (multi-turn agentic RL), `fully_async_policy/` (async training), `one_step_off_policy/`, `teacher_loop/` (distillation), `reward_loop/` (colocated reward model), `vla/` (vision-language-action).

## Running Training (Example)

```bash
python verl/trainer/main_ppo.py \
    actor_rollout_ref.model.path=/path/to/model \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/val.parquet \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    algorithm.adv_estimator=grpo \
    +actor_rollout_ref.rollout.rollout_engine=vllm
```

Training recipes with shell scripts are in `examples/` (e.g., `examples/grpo_trainer/`, `examples/ppo_trainer/`).

## Contribution Policy

See `AGENTS.md` for full AI-assisted contribution rules. Key points:
- Check for duplicate PRs before opening one
- No low-value busywork PRs (single typo, isolated style change)
- Pure code-agent PRs are not allowed; a human must review every line
- Add `Co-authored-by:` trailers for AI-assisted commits
- Do not modify `AGENTS.md` without reading `docs/contributing/editing-agent-instructions.md`
