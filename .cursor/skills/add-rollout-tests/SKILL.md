---
name: add-rollout-tests
description: Design and add tests for veRL rollout and decoupled speculation changes. Use when the user wants to add coverage, generate a test matrix, protect a bug fix, or validate changes in `decoupled_spec_rollout` and related rollout integration code.
---

# Add Rollout Tests

Design tests for rollout and decoupled speculation with explicit coverage goals.

## When To Use

Use this skill when the user asks to:

- add tests for a new rollout feature
- improve coverage
- create regression tests
- check whether a change is sufficiently tested

## Test Planning Rules

Always consider these axes when relevant:

- verify path vs decode fallback path
- aligned draft vs mismatched draft
- full draft length vs short draft length
- eager vs CUDA Graph path
- single request vs batched request
- success path vs missing or delayed draft result

## Recommended Workflow

1. Identify the behavioral contract that must not regress.
2. Decide the cheapest useful test layer:
   - pure unit test
   - mocked integration test
   - subprocess or distributed test
   - benchmark or profiling script
3. Build a compact test matrix.
4. Prioritize must-have tests before nice-to-have tests.
5. If full integration is expensive, add a narrow regression proxy test as well.

## Output Format

1. Behavior under test
2. Test matrix
3. Minimum required tests
4. Suggested test file names
5. Hard-to-test gaps

## Implementation Guidance

- Prefer deterministic tests.
- Mock only at stable boundaries.
- If a bug depends on async timing, state that the test may need synchronization hooks or fake results.
