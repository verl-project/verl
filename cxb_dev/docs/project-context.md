# Project Context

## Purpose

This is the entry document for the current `AgentFramework` / `AgentGateway` workstream on `feature/agent_framework`.

It should let the next agent answer four questions quickly:

1. What is the current architecture baseline?
2. What is already implemented on this branch?
3. What is already settled and should not be re-litigated by default?
4. What still needs follow-up in the next phase?

Detailed design history belongs in `cxb_dev/docs/plans/`, not here.

## Current Phase

Current branch state is a PR1-style experimental slice with the following baseline:

- thin `AgentFramework`
- serving-owned Gateway runtime under `LLMServerManager`
- session-based `/v1/chat/completions` as the main ingress
- `GatewayActor` as the session truth source for chat-completion traffic
- trajectory assembly into training-visible `DataProto`
- minimal OpenAI-compatible reference framework for validation

Not part of this first slice:

- `AgentLoopManager` migration
- token-request ingress
- streaming
- full chat-template / processor parity with existing VERL agent-loop utilities
- broad SWE / Retool / React validation
- trainer-level reward-curve evidence

## Read First

1. Governance: `/home/cxb/rl_framework/verl/cxb_dev/AGENTS.md`
2. Current design baseline: `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-design-v2.md`
3. Current PR1 implementation plan: `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-implementation-plan-v2.md`

Older 2026-03-30 docs are discussion history only, not the current architectural source of truth.

## Frozen Decisions

These are the current maintainer-aligned decisions unless explicitly reopened.

### 1. Thin framework remains the baseline

- `AgentFramework` stays thin.
- Public trainer-facing contract is `generate_sequences(prompts: DataProto) -> DataProto`.
- Do not reintroduce thick public framework hooks such as `run_session` or `compute_reward`.

### 2. Gateway ownership stays on the serving side

- Gateway lifecycle belongs to `LLMServerManager` / serving runtime.
- `GatewayManager` is an internal routing helper, not a top-level framework owner.
- Do not treat old framework-owned Gateway designs as the baseline.

### 3. First-class ingress is session chat completions

- PR1 path is `/v1/chat/completions`.
- Token-request ingress is deferred to later work.

### 4. `AgentLoopManager` migration is deferred

- Existing native agent loops are not the first migration target.
- The first usable path must not depend on moving `AgentLoopManager`.

### 5. Same-session serialization is part of the contract

- Cross-session concurrency remains asynchronous.
- Same-session operations are serialized through one per-session boundary.
- This boundary is not only for chat requests; lifecycle operations matter too.

### 6. `/complete` remains optional

- `POST /sessions/{id}/complete` is important for remote / hosted agents.
- It is not mandatory for every local completion path.

## Current Code Facts

This branch already contains a substantial experimental implementation.

### Framework side

- `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/framework.py`
  - thin abstract `AgentFramework` with only `generate_sequences`
- `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/openai_compatible_framework.py`
  - minimal reference framework that creates sessions, runs an agent, optionally waits, and finalizes
- `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/assembler.py`
  - `TrajectoryAssembler` that converts trajectories into training-visible `DataProto`

### Gateway side

- `/home/cxb/rl_framework/verl/verl/experimental/agent_gateway/gateway.py`
  - `GatewayActor` now has explicit session phases and lifecycle guards
  - duplicate session creation is rejected
  - malformed request and unsupported-but-well-formed request are distinguished
  - canonical request context tracks `messages + tools`
  - assistant `tool_calls`, `tool_call_id`, and multimodal `content` are retained in canonical state
  - `name` is currently treated as unsupported in PR1
  - same-session chat / complete / finalize / abort are serialized
  - backend failure follows no-commit semantics
- `/home/cxb/rl_framework/verl/verl/experimental/agent_gateway/manager.py`
  - sticky session routing plus least-active-session placement

### Serving integration

- `/home/cxb/rl_framework/verl/verl/experimental/agent_loop/agent_loop.py`
  - `LLMServerManager` owns optional Gateway runtime and exposes session runtime methods
  - `AsyncLLMServerManager` provides the serving-backed Gateway backend path

### Tests already present

- `/home/cxb/rl_framework/verl/tests/experimental/agent_framework/test_assembler.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_framework/test_reward_helpers.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_framework/test_openai_compatible_framework.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_gateway_actor.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_gateway_manager.py`
- `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_session_runtime.py`

## What Is Already Resolved

These items were previously open and should not be reintroduced as unresolved by default:

- explicit minimal session state model is now part of the design direction
- duplicate session overwrite is no longer acceptable behavior
- unsupported fields should not be silently ignored just to look OpenAI-compatible
- `tools` belongs to the request compatibility boundary
- prefix continuation and mismatch semantics should be based on canonical request context, not text-only flattening
- backend failure should not partially commit session state

## Main Remaining Risks

Only keep the current phase's real follow-up items here.

### 1. Chat-template / processor parity is still the biggest correctness risk

Current Gateway implementation still uses a simplified `tokenizer.encode_messages(...)` path in its runtime and tests. That is not the same as VERL's existing `apply_chat_template(...)`, `initialize_system_prompt(...)`, `process_vision_info(...)`, and processor-driven multimodal path.

This is not just a documentation concern; it is a real implementation gap that may affect correctness once the Gateway path is expected to behave like native VERL tool or multimodal flows.

### 2. Response-side completion path still needs assessment

PR1 tightened request-side truth source and lifecycle semantics, but the next phase still needs to assess whether completion-driven agents need additional response-side structure, such as tool-call parsing helpers or stronger assistant-message contracts.

### 3. Validation scope is still narrow

Current validation is strong at the unit/component level, but still narrow in representativeness:

- minimal mock remote-style path exists
- serving ownership path exists
- broad SWE / Retool / React validation does not yet exist

## Recommended Next Step

Do not reopen the settled PR1 GatewayActor debates first.

Instead, move to the next-phase planning questions:

1. assess chat-template alignment feasibility and risk
2. assess completion return-path completeness for tool-using agents
3. assess whether SWE-agent is worth adapting into the completion path as a main validation target
4. run one more independent review pass for remaining fixups and doc/impl drift

The working note for that next phase is:

- `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-03-agent-framework-review-handoff.md`

## Fast Onboarding Order

If someone is starting fresh, use this order:

1. `/home/cxb/rl_framework/verl/cxb_dev/AGENTS.md`
2. `/home/cxb/rl_framework/verl/cxb_dev/docs/project-context.md`
3. `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-design-v2.md`
4. `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-implementation-plan-v2.md`
5. `/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-03-agent-framework-review-handoff.md`
6. `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_gateway_actor.py`
7. `/home/cxb/rl_framework/verl/verl/experimental/agent_gateway/gateway.py`
8. `/home/cxb/rl_framework/verl/tests/experimental/agent_gateway/test_session_runtime.py`
9. `/home/cxb/rl_framework/verl/verl/experimental/agent_loop/agent_loop.py`
10. `/home/cxb/rl_framework/verl/tests/experimental/agent_framework/test_openai_compatible_framework.py`
