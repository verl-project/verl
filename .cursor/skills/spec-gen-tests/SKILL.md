---
name: spec-gen-tests
description: Generate rollout and decoupled speculation test plans or test code skeletons. Invoke explicitly with `/spec-gen-tests` when the user wants concrete tests for a rollout change.
disable-model-invocation: true
---

# Spec Gen Tests

Use this skill only when explicitly invoked as `/spec-gen-tests`.

## Purpose

Generate high-value tests for `decoupled_spec_rollout` and related rollout code.

## Workflow

1. Identify the changed behavior.
2. Map it to the smallest useful test layer.
3. Produce:
   - suggested test files
   - test case list
   - critical assertions
   - code skeletons if requested

## Priority Cases

- verify path
- decode fallback path
- mismatched first token handling
- short draft vs full draft
- graph-enabled vs graph-disabled behavior

## Output Format

1. Suggested files
2. Test cases
3. Assertions to protect
4. Optional code skeleton
