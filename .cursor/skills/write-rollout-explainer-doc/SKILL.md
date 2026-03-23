---
name: write-rollout-explainer-doc
description: Write explainer documents for veRL rollout and decoupled speculation internals. Use when the user wants architecture notes, onboarding docs, code walkthroughs, or implementation explanations for `decoupled_spec_rollout`, `sglang_rollout`, and related patches.
---

# Write Rollout Explainer Doc

Turn implementation details into documentation that another engineer can quickly use.

## Documentation Goals

- explain why the module exists
- map the main components and responsibilities
- show how a typical request flows through the system
- surface common misunderstandings
- provide debugging entrypoints

## Preferred Audience

This skill is especially useful for:

- teammates new to decoupled speculation
- engineers familiar with veRL but not SGLang or vLLM internals
- reviewers who need design context

## Recommended Structure

1. Background
2. Main components
3. Request flow
4. Key branch conditions
5. Gotchas
6. Debugging tips

## Writing Rules

- Prefer exact file and symbol references over vague descriptions.
- Explain local patches separately from upstream behavior.
- If behavior differs between SGLang and vLLM, compare them directly.
- Favor internal design-doc style over tutorial fluff.

## Output Format

Use clean markdown that can be copied into a design note or internal documentation page.
