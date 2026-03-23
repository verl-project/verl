---
name: trace-rollout-dataflow
description: Trace request flow, state transitions, and data structures across veRL rollout and decoupled speculation code. Use when the user asks how a rollout path works, how data moves between draft and verifier, or how a token, batch, or result is transformed across `decoupled_spec_rollout`, `sglang_rollout`, `sglang-v0.5.9`, and `vllm_4_specrl`.
---

# Trace Rollout Dataflow

Trace a rollout path from entrypoint to output.

## When To Use

Use this skill when the user asks:

- "这条链路怎么走"
- "这段逻辑是什么意思"
- "DraftResult / VerifyInput / GenerationBatchResult 是怎么变化的"
- "为什么这里会走 verify / decode / fallback"

## Core Instructions

1. Start from the concrete entrypoint file or symbol if the user provides one.
2. Identify the request lifecycle across these layers when relevant:
   - rollout orchestration
   - decoupled speculation glue code
   - SGLang or vLLM executor behavior
3. Track the key data structures explicitly.
4. For each branch, state:
   - trigger condition
   - input state
   - output state
   - side effects
5. Distinguish semantic flow from optimization details such as CUDA Graph.

## Output Format

Use this structure:

1. Conclusion
2. Entry point
3. Step-by-step flow
4. Branch conditions
5. Important state mutations
6. Easy-to-miss details

## Checklist

- Mention exact file paths and symbols.
- Call out where request state is mutated in place.
- If multiple runtimes are involved, say which layer owns the final behavior.
