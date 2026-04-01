# CLAUDE.md - veRL

## WHAT: Project Overview

veRL (Volcano Engine Reinforcement Learning) is a flexible, efficient, production-ready
RL training library for LLMs, initiated by ByteDance Seed.

**Tech Stack**: Python 3.10+ | PyTorch | FSDP / Megatron-LM | vLLM / SGLang | Ray

**Core Directories**:

- `verl/` - Core package
  - `trainer/` - PPO trainer, core algorithm utilities (advantage estimators, policy loss)
  - `workers/` - Ray remote workers (actor, critic, rollout, reward_manager, sharding_manager)
  - `utils/` - Shared utilities
    - `reward_score/` - `compute_score` functions (one per task/dataset)
    - `dataset/` - `RLDataset` and preprocessing utilities
    - `vllm/`, `sglang/` - Inference engine integration
    - `megatron/`, `fsdp_utils.py` - Training backend utilities
  - `protocol.py` - `DataProto`: the core data container between controller and workers
  - `models/` - Model registry and weight loaders
- `examples/` - Trainer recipes and data preprocessing scripts
  - `data_preprocess/` - Dataset conversion scripts (raw → parquet)
  - `grpo_trainer/`, `ppo_trainer/`, `dapo/`, `rloo_trainer/`, etc. - Algorithm recipes
- `recipe/` - Community-contributed training recipes (git submodule)
- `tests/` - Unit and integration tests

## WHY: Purpose

- Efficient RL training for LLM post-training (RLHF, RLVR) at scale
- HybridEngine: same GPU pool for training (FSDP/Megatron) and inference (vLLM/SGLang)
  — eliminates memory redundancy during weight sync
- Modular: reward functions, datasets, and trainer logic are independently extensible

## HOW: Core Commands

```bash
# Install dependencies
pip install -e ".[gpu]"
# or with uv:
uv pip install -e ".[gpu]"

# Preprocess a dataset (run once offline)
python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

# Launch training (local single-node)
bash examples/grpo_trainer/run_qwen2-7b_math.sh

# Run tests
pytest tests/

# Format and lint
pre-commit run --all-files
```

## Boundaries

### Constraints

- Designed for multi-GPU / multi-node Ray clusters
- Integration tests require GPUs; explain skips when unavailable
- `recipe/` is a git submodule — run `git submodule update --init --recursive` first

### Always Do

- Read relevant files before modifying code
- Follow existing code patterns in the same module
- Add `compute_score` to `verl/utils/reward_score/__init__.py` dispatch table when
  adding a new reward function
- Set `data_source` in preprocessed dataset to match the reward dispatch key

### Ask First

- Modifying `verl/protocol.py` (`DataProto` schema changes affect all workers)
- Adding new dependencies
- Changing worker placement or Ray resource allocation logic
- Renaming public APIs used in `examples/`

### Never Do

- Hardcode file paths or cluster-specific endpoints
- Skip pre-commit hooks
- Use wildcard imports (`from x import *`)
- Raise exceptions inside `compute_score` (return `0.0` instead)

## PR Workflow (contributing to upstream)

Every upstream PR **must branch from `upstream/main`**, not local `main`:

```bash
git fetch upstream main
git checkout -b feat/xxx upstream/main
# make targeted changes
git add <specific files>
git commit -m "feat(scope): description"
git push origin feat/xxx
gh pr create --repo verl-project/verl --head <your-fork>:feat/xxx --base main
```

Verify diff before opening: `git diff upstream/main`

## Progressive Disclosure: Skills

| Task                    | Skill                  |
| ----------------------- | ---------------------- |
| Add reward function     | `/add-reward`          |
| Add dataset             | `/add-dataset`         |
| Add new trainer/recipe  | `/add-trainer`         |
| Add unit tests          | `/add-unit-tests`      |
| Debug distributed issue | `/debug-distributed`   |
| Create / update PR      | `/create-pr`           |
| Review a PR             | `/review-pr`           |
| Upgrade vLLM version    | `/upgrade-vllm`        |
| Upgrade SGLang version  | `/upgrade-sglang`      |
| Upgrade Megatron-Core   | `/upgrade-megatron-core` |
| Commit message format   | `/commit-conventions`  |

Skills live in `.agents/skills/`.

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), ≤72 chars subject,
  imperative voice, reasoning in body
- **Squash**: Squash WIP commits before opening PR
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations
