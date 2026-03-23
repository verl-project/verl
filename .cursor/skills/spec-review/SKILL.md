---
name: spec-review
description: Review decoupled speculation and rollout changes with emphasis on correctness, async state handling, fallback behavior, CUDA Graph assumptions, and test gaps. Invoke explicitly with `/spec-review`.
disable-model-invocation: true
---

# Spec Review

Use this skill only when explicitly invoked as `/spec-review`.

## Review Focus

- correctness of verifier and draft interaction
- async ordering and stale state risk
- fallback decode semantics
- graph capture and replay assumptions
- protocol and state mutation safety
- missing regression tests

## Review Workflow

1. Identify the changed paths.
2. Trace behavior before and after the change.
3. Look for branch-specific regressions.
4. Separate correctness findings from performance concerns.
5. End with concrete test suggestions.

## Output Format

1. Findings ordered by severity
2. Open questions or assumptions
3. Brief overall assessment
4. Test coverage gaps
