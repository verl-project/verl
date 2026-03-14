# veRL Multi-Turn VLM RL Training Project (Codex Guide)

This file is the Codex-oriented companion to `CLAUDE.md`. It keeps the same project context and workflows, but is written as an execution guide for Codex agents.

## Project Overview

This repository is a veRL fork focused on multi-turn VLM reinforcement learning for GUI agent tasks.

Key focus areas:
- Multi-turn dialogue training with tool-calling
- Vision-language RL for GUI tasks
- Distributed training with Ray + FSDP/FSDP2

## Working Principles

- Minimal implementation first: solve only the requested scope.
- Reuse existing patterns: prefer local consistency over new abstractions.
- No unsolicited cleanup: avoid unrelated refactors while fixing a targeted issue.

## Runtime Requirements

All veRL/GPU training and test commands should run inside the verl Docker container.

```bash
docker exec -it verl bash
# or
docker exec verl <command>
```

Before any GPU workload, check whether GPUs are currently occupied:

```bash
docker exec verl nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Rules:
- No active process: safe to run GPU tests.
- Active process found: do not start GPU jobs until confirmed.

## RL Training Critical Path

Understanding this path is required for debugging training to sandbox interactions:

```
verl/utils/dataset/rl_dataset.py
  -> verl/trainer/ppo/ray_trainer.py
    -> verl/experimental/agent_loop/agent_loop.py
      -> verl/experimental/agent_loop/gui_agent_loop.py
        -> verl/tools/os_sandbox_tool.py
          -> HTTP boundary
            -> sandbox proxy_service / adapter / executor / VM manager
```

Cross-repo boundary:
- Above `os_sandbox_tool.py`: GPU training side.
- Below proxy service: sandbox execution side.

## Verification-First Checklist

Before implementation, define validation up front:
- Unit tests to add or update
- Boundary/edge cases to cover
- Baseline comparison method
- End-to-end check for behavior and performance

## Experiment Economics

Experiments are expensive; analysis is cheap.

- Always predict expected outcomes before running experiments.
- Stop and analyze after two unchanged experiment results.
- Use waiting time to inspect the next stage of code paths.
- For memory/throughput issues, estimate theoretical limits first, then validate.

## Git Workflow Expectations

- Confirm intent before git operations: `commit only` vs `commit + push` vs `create/update PR`.
- Use atomic Conventional Commits.
- Do not commit secrets, large binaries, ad-hoc debug artifacts, or unrelated file noise.

## Key Files

Entry points:
- `verl/trainer/main_ppo.py`
- `multi_turn_baseline.sh`

Core logic:
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/reward.py`

Workers and execution:
- `verl/workers/fsdp_workers.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/rollout/`

Config and data:
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/engine/`
- `verl/utils/dataset/rl_dataset.py`
- `verl/protocol.py`

## References

- veRL docs: https://verl.readthedocs.io/
- Hydra docs: https://hydra.cc/docs/intro/
- Ray docs: https://docs.ray.io/
- Source companion: `CLAUDE.md`
