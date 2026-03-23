---
name: spec-upgrade-check
description: Assess upgrade and patch impact for decoupled speculation dependencies such as SGLang and vLLM. Invoke explicitly with `/spec-upgrade-check` when the user wants to evaluate upstream changes, local patch assumptions, or migration risk.
disable-model-invocation: true
---

# Spec Upgrade Check

Use this skill only when explicitly invoked as `/spec-upgrade-check`.

## Purpose

Evaluate the impact of changing upstream code that decoupled speculation depends on.

## Typical Targets

- `sglang-v0.5.9`
- `vllm_4_specrl`
- `verl/workers/rollout/sglang_rollout`
- local patch files under `decoupled_spec_rollout/sglang_patch`

## Analysis Checklist

1. Which local assumptions depend on current upstream behavior
2. Which patched symbols or paths are likely to break
3. Which behaviors are semantic versus incidental
4. What minimum validation is required after upgrade

## Output Format

1. Impact summary
2. High-risk assumptions
3. Files most likely needing updates
4. Validation checklist
5. Recommended migration order
