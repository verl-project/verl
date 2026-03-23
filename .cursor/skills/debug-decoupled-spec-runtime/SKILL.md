---
name: debug-decoupled-spec-runtime
description: Diagnose runtime issues in veRL decoupled speculation. Use when the user reports wrong outputs, hangs, stale results, verifier and draft mismatch, async ordering problems, or unexpected fallback behavior in `decoupled_spec_rollout`.
---

# Debug Decoupled Spec Runtime

Debug runtime issues in decoupled speculation with a structured approach.

## Typical Triggers

- wrong token output
- verifier and draft disagree
- first token mismatch behavior is unclear
- stale proxy result
- async path behaves differently from sync path
- runtime only fails when CUDA Graph is enabled

## Debug Workflow

1. Clarify the symptom:
   - wrong result
   - missing result
   - hang or timeout
   - performance cliff
2. Find the earliest divergence point.
3. Trace request state across:
   - draft result production
   - verify input construction
   - verifier forward path
   - result post-processing
   - follow-up draft triggering
4. Build 2-3 ranked hypotheses.
5. Recommend the smallest next check to confirm or eliminate each hypothesis.

## Areas To Inspect

- `protocol.py`
- `draft_proxy.py`
- `verify_server_patch.py`
- `decoupled_spec_verify_patch.py`
- `async_sglang_server.py`
- any touched upstream patch path in `sglang-v0.5.9` or `vllm_4_specrl`

## Output Format

1. Symptom summary
2. Most likely bug points
3. Evidence and reasoning
4. Debug logs or assertions to add
5. Likely fix layer

## Special Rules

- Separate correctness bugs from logging gaps.
- Be explicit about whether the issue is pre-verify, verify-time, or post-verify.
- If CUDA Graph changes the symptom, call out whether it changes semantics or only execution path.
