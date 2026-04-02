# Project Context

## Purpose

This is the entry document for the current `AgentFramework` / `AgentGateway` workstream on `feature/agent_framework`.

The next agent should be able to answer three questions quickly:

1. What is the current architecture baseline?
2. What is already implemented on this branch?
3. What still needs careful follow-up?

Detailed design history belongs in `cxb_dev/docs/plans/`.

## Current Phase

Current branch state is an experimental PR1-style slice:

- thin `AgentFramework`
- serving-owned Gateway runtime
- `/v1/chat/completions` session path
- trajectory assembly into training-visible `DataProto`
- minimal OpenAI-compatible reference framework

Not in the current first slice:

- `AgentLoopManager` migration
- token-request ingress
- fully async integration
- React / Retool / SWE validation
- end-to-end reward-curve evidence

## Canonical References

Read these first:

1. RFC: `/home/cxb/MATE-reboot/docs/rfc/agentFramework_agentgateway_rfc/agentFramework_agentgateway_rfc.md`
2. Governance: `/home/cxb/rl_framework/verl/cxb_dev/AGENTS.md`
3. Current design baseline: `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-design-v2.md`
4. Current implementation plan: `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-implementation-plan-v2.md`

Older 2026-03-30 docs are discussion history only. They are useful for a few Gateway behavior details and old test ideas, but they are not the architectural source of truth anymore.

## Frozen Decisions

These are the current maintainer-aligned decisions unless explicitly reopened.

### 1. Thin framework

- `AgentFramework` stays thin.
- Public trainer-facing contract is `generate_sequences(prompts: DataProto) -> DataProto`.
- `run_session`, `compute_reward`, `finalize_session` are not required public framework interfaces.

### 2. Gateway ownership is on the serving side

- Gateway lifecycle belongs to `LLMServerManager` / serving runtime.
- `GatewayManager` is an internal routing/control-plane helper, not a top-level framework owner.
- Do not resurrect `AgentFrameworkManager -> FrameworkWorker -> AgentFramework` as the public architecture.

### 3. First-class ingress is chat completions

- PR1 path is `/v1/chat/completions`.
- Token-request ingress is deferred for later `AgentLoopManager` migration.

### 4. `AgentLoopManager` migration is deferred

- Current `AgentLoopManager` is not the first migration target.
- The first usable path must not depend on migrating existing native agent loops.

### 5. Gateway concurrency semantics are in scope

- Cross-session concurrency should remain asynchronous.
- Same-session requests should be serialized with a per-session lock.
- Same-session reentry should wait rather than fail with `409/429`.

### 6. `/complete` remains optional

- `POST /sessions/{id}/complete` is important for remote / hosted agents.
- It should not be mandatory for every local OpenAI-compatible path.

## Current Code State

This branch already contains a substantial experimental implementation.

### Framework side

- `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/framework.py`
  - `AgentFramework` is now a thin abstract base with only `generate_sequences`.
- `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/types.py`
  - lightweight `SessionHandle` and `Trajectory` types
- `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/helpers.py`
  - reward normalization and trajectory validation helpers
- `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/assembler.py`
  - `TrajectoryAssembler` for training-visible `DataProto`
- `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/openai_compatible_framework.py`
  - minimal reference implementation for an OpenAI-compatible / remote-style execution model

### Gateway side

- `/home/cxb/rl_framework/verl/verl/experimental/agent_gateway/gateway.py`
  - `GatewayActor` with:
    - `/sessions/{id}/v1/chat/completions`
    - `/sessions/{id}/complete`
    - trajectory materialization on prefix mismatch / finalize
    - per-session `asyncio.Lock`
    - session-state inspection fields (`metadata`, completion flags, timestamps)
- `/home/cxb/rl_framework/verl/verl/experimental/agent_gateway/manager.py`
  - `GatewayManager` with sticky routing and least-active-session placement

### Serving integration

- `/home/cxb/rl_framework/verl/verl/experimental/agent_loop/agent_loop.py`
  - `LLMServerManager` now owns optional Gateway runtime and exposes:
    - `create_session`
    - `finalize_session`
    - `abort_session`
    - `wait_for_completion`
  - `AsyncLLMServerManager` provides the serving-backed Gateway backend path

### Tests already present

- `/home/cxb/rl_framework/verl/tests/experimental/agent_framework/test_assembler.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_framework/test_reward_helpers.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_framework/test_openai_compatible_framework.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_gateway_actor.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_gateway_manager.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_session_runtime.py`

## Useful Reference-Branch Findings

The archived `cxb/agent-framework-gateway` branch is only a reference source for behavior details and missed test ideas.

The most credible carry-over candidates are:

- richer `/v1/chat/completions` normalization and validation:
  - `name`
  - `tool_calls`
  - `tools`
  - malformed-request 4xx behavior
- duplicate-session rejection on `create_session`
- one regression test that locks in prefix-match continuation as a single trajectory with mixed `response_mask` semantics
- trace/debug metadata propagation questions such as whether `uid`, sample index, and `trajectory_id` should survive into assembled `DataProto`

Do not carry over from that branch:

- `AgentFrameworkManager`
- `FrameworkWorker`
- thick framework hooks such as public `run_session` / `compute_reward`
- framework-side gateway ownership
- round-robin routing as the preferred policy

## Open Questions And Risks

These are the main points still worth checking before treating this slice as ready for broader review.

### 1. Chat-template alignment

Current Gateway implementation on this branch uses a simplified `tokenizer.encode_messages(...)` path in tests and implementation. That is not yet aligned with VERL's existing `apply_chat_template` / `initialize_system_prompt` utilities.

This is the biggest current correctness risk if PR1 is expected to match VERL-native chat templating behavior.

### 2. OpenAI compatibility level

The branch has the main chat-completions path, but the intended compatibility boundary is still worth making explicit:

- whether `tools` / `tool_calls` / `name` must be supported in PR1
- how strict malformed-request responses should be
- whether duplicate session creation should raise a clear serving-side error

### 3. Metadata propagation contract

`Trajectory` already contains `uid`, `session_id`, and `trajectory_id`, but current assembler output mainly guarantees training tensors plus `__num_turns__` and extra reward fields.

If traceability across session -> trajectory -> batch matters for debugging or evaluation, decide whether those ids must be preserved in assembled `DataProto`.

### 4. Validation scope is still limited

Current branch has focused component tests and a minimal remote-style path, but not yet:

- trainer-level end-to-end evidence
- reward-curve evidence
- `AgentLoopManager` compatibility migration
- fully async validation

## Recommended Next Steps

1. Keep the current thin-framework / serving-owned-Gateway architecture unchanged.
2. Review whether PR1 should absorb a small subset of archived Gateway behavior:
   - duplicate-session rejection
   - better malformed-request handling
   - prefix-match regression coverage
3. Decide explicitly whether chat-template alignment with VERL utilities is required in PR1 or deferred.
4. Decide explicitly whether `uid` / `session_id` / `trajectory_id` must survive into assembled `DataProto`.
5. If implementation continues, prioritize verification over expansion:
   - session lifecycle correctness
   - compatibility boundary documentation
   - one trainer-visible acceptance check

## Fast Onboarding Order

1. `/home/cxb/MATE-reboot/docs/rfc/agentFramework_agentgateway_rfc/agentFramework_agentgateway_rfc.md`
2. `/home/cxb/rl_framework/verl/cxb_dev/AGENTS.md`
3. `/home/cxb/rl_framework/verl/cxb_dev/docs/project-context.md`
4. `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-design-v2.md`
5. `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-implementation-plan-v2.md`
6. `/home/cxb/rl_framework/verl/verl/experimental/agent_loop/agent_loop.py`
7. `/home/cxb/rl_framework/verl/verl/experimental/agent_gateway/gateway.py`
8. `/home/cxb/rl_framework/verl/verl/experimental/agent_gateway/manager.py`
9. `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/framework.py`
10. `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/openai_compatible_framework.py`
11. `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_gateway_actor.py`
12. `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_session_runtime.py`
