---
name: review-pr
description: Read-only pull request review for veRL with risk analysis and targeted checklists.
---

# Review Pull Request

Use this skill when the user asks for a PR review of the current branch or a specific PR.

## Inputs

- Optional PR number (defaults to current branch's PR)
- Optional `--quick` to stop after the change analysis phase

## Hard Rules

- **Stay read-only.** Do not edit files, commit, push, rebase, or change GitHub state.
- Do not run build, install, or test commands that mutate the environment.
- Use `gh` for PR metadata and `git diff` for content.

## Workflow

### Phase 1: Resolve PR Context

```bash
gh pr view [<number>] --json number,title,body,state,isDraft,files,baseRefName,headRefName
```

- If no PR exists or it is closed, stop and report clearly.
- Record: branch name, changed files, PR title format compliance.

**veRL PR title format check** (enforced by CI):

```
[{modules}] {type}: {description}
```

Flag if title doesn't match this pattern.

### Phase 2: Change Analysis

Classify changed files into risk areas:

| Risk Area | Paths | Risk Level |
|-----------|-------|------------|
| Protocol / DataProto | `verl/protocol.py` | CRITICAL |
| HybridEngine / sharding | `verl/workers/sharding_manager/` | CRITICAL |
| Core algorithm (loss, advantage) | `verl/trainer/ppo/core_algos.py` | HIGH |
| Workers (actor, critic, rollout) | `verl/workers/actor/`, `critic/`, `rollout/` | HIGH |
| Reward manager | `verl/workers/reward_manager/` | HIGH |
| vLLM / SGLang integration | `verl/utils/vllm/`, `verl/utils/sglang/` | HIGH |
| Trainer orchestration | `verl/trainer/` | MEDIUM |
| Dataset / reward_score | `verl/utils/dataset/`, `verl/utils/reward_score/` | MEDIUM |
| Config / base_config | `verl/base_config.py`, `verl/workers/config/` | MEDIUM |
| Examples / recipes | `examples/`, `recipe/` | LOW |
| Docs | `docs/` | LOW |
| CI | `.github/workflows/` | LOW |

Build a `CHANGE_ANALYSIS_REPORT`:
- Detected risk areas
- Highest risk level: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`
- Affected files
- Likely failure modes

If `--quick`, return the report and stop.

### Phase 3: Review Planning

Select review passes based on risk areas. Always include at least one general logic pass.
Split by risk area, not by file count.

### Phase 4: Execute Review Passes

For each risk area, check the following:

#### CRITICAL: Protocol / DataProto changes
- Does `DataProto` schema change break downstream workers?
- Are all `.batch`, `.non_tensor_batch`, `.meta_info` accesses updated consistently?
- Does the change affect checkpoint compatibility?

#### CRITICAL: HybridEngine / Sharding
- Are training TP and rollout TP sizes still compatible?
- Is NCCL group initialization correct across all placement strategies?
- Is memory released properly between training and inference phases?

#### HIGH: Algorithm (core_algos.py)
- Is `response_mask` applied correctly before loss/advantage computation?
- Is advantage normalization level (`batch` vs `group`) consistent with the algorithm?
- Are importance ratios clipped correctly (old_log_probs alignment with new_log_probs)?
- Does NaN propagation from `log(0)` get handled?

#### HIGH: Workers
- Are Ray remote calls properly awaited (`.remote()` + `ray.get()`)?
- Is GPU memory released between rollout and training phases?
- Are collective ops (all_reduce, broadcast) called on matching ranks?

#### HIGH: vLLM / SGLang integration
- Do imported vLLM symbols still exist at the pinned version (`requirements.txt`)?
- Are try/except fallbacks for version-specific imports still correct?
- Is `SamplingParams` usage compatible with the pinned vLLM version?

#### MEDIUM: Reward / Dataset
- Does `compute_score` handle exceptions and return `0.0` (never raise)?
- Does `data_source` in preprocessed dataset match the dispatch key in `__init__.py`?
- Are all required DataProto fields present in dataset output?

#### MEDIUM: Config changes
- Does renaming a config field break existing YAML run scripts in `examples/`?
- Is backward compatibility handled or `[BREAKING]` added to PR title?

### Phase 5: Final Report

```markdown
CHANGE_ANALYSIS_REPORT:
- risk_level: CRITICAL | HIGH | MEDIUM | LOW
- detected_risk_areas: [...]
- affected_files: [...]
- likely_failure_modes: [...]

Findings (ordered by severity)
1. [CRITICAL] Title — path:line
   - Problem: ...
   - Fix: ...

2. [HIGH] ...

Open Questions
- ...

Residual Risk
- ...

PR Format Check
- Title compliant: yes/no
- pre-commit: (can't verify, remind reviewer)
- Tests added: yes/no (based on diff)
- Docs updated: yes/no (based on diff)
```

## What to Ignore

- Pure style nits already caught by pre-commit (ruff, isort)
- Issues outside the changed scope unless the PR makes them worse
- Speculative concerns with no concrete trigger in the diff

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/review-pr/SKILL.md

## How to Update
- When new high-risk modules added: update Phase 2 risk table
- When PR title format changes (check-pr-title.yml): update Phase 1
================================================================================
-->
