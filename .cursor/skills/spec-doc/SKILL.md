---
name: spec-doc
description: Generate concise explainer or design documentation for decoupled speculation code paths. Invoke explicitly with `/spec-doc` when the user wants a readable document for teammates or future maintainers.
disable-model-invocation: true
---

# Spec Doc

Use this skill only when explicitly invoked as `/spec-doc`.

## Purpose

Write documentation for rollout and decoupled speculation internals that is easier to consume than raw code.

## Typical Outputs

- architecture note
- onboarding doc
- module explainer
- request flow walkthrough
- debugging guide

## Default Structure

1. Background
2. Main components
3. Request flow
4. Important branch conditions
5. Common misunderstandings
6. Debugging entrypoints

## Rules

- Prefer exact paths and symbols.
- Explain why, not just what.
- Separate local patch behavior from upstream behavior.
