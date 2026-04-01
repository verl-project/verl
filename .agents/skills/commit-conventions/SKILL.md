---
name: commit-conventions
description: veRL commit message conventions. Load on every git commit to enforce Conventional Commits format.
---

# Commit Conventions

veRL follows the **Conventional Commits** specification.

## Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

- **Subject**: ≤72 chars, imperative mood ("add", "fix", not "added", "fixes")
- **Body**: explain *why*, not *what*; wrap at 72 chars
- **Footer**: reference issues (`Closes #123`, `Fixes #456`)

## Types

| Type       | When to Use                                              |
| ---------- | -------------------------------------------------------- |
| `feat`     | New feature or algorithm                                 |
| `fix`      | Bug fix                                                  |
| `perf`     | Performance improvement (no behavior change)             |
| `refactor` | Code restructuring (no feature/fix)                      |
| `docs`     | Documentation only                                       |
| `test`     | Adding or fixing tests                                   |
| `chore`    | Build, CI, dependencies, tooling                         |
| `revert`   | Reverting a previous commit                              |

## Scopes (inferred from file paths)

| Scope        | Paths                                           |
| ------------ | ----------------------------------------------- |
| `trainer`    | `verl/trainer/`                                 |
| `workers`    | `verl/workers/`                                 |
| `reward`     | `verl/utils/reward_score/`, `workers/reward_manager/` |
| `dataset`    | `verl/utils/dataset/`                           |
| `rollout`    | `verl/workers/rollout/`                         |
| `actor`      | `verl/workers/actor/`                           |
| `critic`     | `verl/workers/critic/`                          |
| `fsdp`       | `verl/workers/fsdp_workers.py`, `verl/utils/fsdp_utils.py` |
| `megatron`   | `verl/workers/megatron_workers.py`, `verl/utils/megatron/` |
| `vllm`       | `verl/utils/vllm/`                              |
| `sglang`     | `verl/utils/sglang/`                            |
| `protocol`   | `verl/protocol.py`                              |
| `recipe`     | `recipe/`                                       |
| `examples`   | `examples/`                                     |
| `ci`         | `.github/`                                      |

## Examples

```
feat(reward): add process-level reward scoring for code tasks

fix(trainer): fix NaN loss when response_mask is all zeros

perf(rollout): reduce memory copies during sequence packing

docs(recipe): add DAPO recipe README with benchmark results

chore(ci): update torch version to 2.5.1 in test matrix
```

## PR Workflow (upstream contribution)

When submitting to `verl-project/verl`, always branch from `upstream/main`:

```bash
git fetch upstream main
git checkout -b feat/my-feature upstream/main
# ... make changes ...
git add <specific files>
git commit -m "feat(scope): description"
git push origin feat/my-feature
gh pr create --repo verl-project/verl --head <your-fork>:feat/my-feature --base main
```

Verify diff before opening PR:
```bash
git diff upstream/main  # must only contain intended changes
```

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/commit-conventions/SKILL.md
Loaded: automatically on every git commit

## How to Update
- Add scopes when new major directories are added
- Keep type list stable (Conventional Commits spec)
================================================================================
-->
