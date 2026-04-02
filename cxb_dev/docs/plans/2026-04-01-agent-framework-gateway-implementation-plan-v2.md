# AgentFramework And AgentGateway Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
>
> This plan updates and supersedes the previous contents of `2026-04-01-agent-framework-gateway-implementation-plan-v2.md` using the latest design baseline in `2026-04-01-agent-framework-gateway-design-v2.md`.

**Goal:** Implement the PR1 AgentGateway path around session-based `/v1/chat/completions`, with `GatewayActor` aligned to the latest lifecycle, canonical request context, and failure semantics.

**Architecture:** Keep `AgentFramework` thin and keep Gateway ownership under `LLMServerManager`. Implement `GatewayActor` as the session truth source for chat-completion traffic, using canonical structured request context instead of text-only normalization, and enforce no-commit behavior on request failure. Reuse VERL chat-template and multimodal helper boundaries where practical, but do not pull `AgentLoopManager` into the PR1 execution path.

**Tech Stack:** Python, FastAPI, Ray actors, pytest, httpx, VERL agent framework / gateway / agent loop modules.

---

## Series Scope

- In scope for this plan:
  - `GatewayActor` lifecycle and state guards
  - duplicate session handling
  - canonical structured request handling for `messages + tools`
  - OpenAI-compatible validation boundary for PR1
  - no-commit / failure semantics
  - `GatewayManager` and `LLMServerManager` integration updates required by the above
  - regression tests covering the new contracts
- Explicitly out of scope:
  - token ingress
  - `AgentLoopManager` migration
  - streaming
  - full `name` semantics
  - broad SWE/Retool validation expansion

### Task 1: Lock Lifecycle and Duplicate Session Behavior

**Files:**
- Modify: `verl/experimental/agent_gateway/gateway.py`
- Modify: `verl/experimental/agent_gateway/types.py`
- Test: `tests/experimental/agent_gateway/test_gateway_actor.py`

**Step 1: Write the failing tests**

Add or update tests covering:

- duplicate `create_session(session_id)` is rejected instead of silently overwriting
- `complete_session` transitions `ACTIVE -> COMPLETED`
- `finalize_session` is allowed from `ACTIVE` and `COMPLETED`
- `finalize_session` fails on `FINALIZED` and `ABORTED`
- `abort_session` is idempotent on `ABORTED` and fails on `FINALIZED`
- `wait_for_completion` succeeds on `COMPLETED`, fails on `ABORTED`, and times out on `ACTIVE`

Suggested test skeleton:

```python
async def test_gateway_actor_rejects_duplicate_session(ray_runtime):
    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["OK"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    ray.get(actor.create_session.remote("dup"))
    with pytest.raises(ray.exceptions.RayTaskError, match="dup"):
        ray.get(actor.create_session.remote("dup"))
```

**Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py -k "duplicate or complete or finalize or abort or wait" -v
```

Expected:

- at least one failure because duplicate create is currently allowed or lifecycle guards are incomplete

**Step 3: Implement the minimal lifecycle changes**

Implement in `verl/experimental/agent_gateway/gateway.py` and `verl/experimental/agent_gateway/types.py`:

- explicit minimal session phase tracking: `ACTIVE`, `COMPLETED`, `FINALIZED`, `ABORTED`
- duplicate create rejection
- lifecycle guards for `chat_completions`, `complete_session`, `finalize_session`, `abort_session`
- `wait_for_completion` terminal behavior aligned with the design doc
- terminal outcome signaling before runtime removal so waiters do not hang

**Step 4: Re-run the focused tests**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py -k "duplicate or complete or finalize or abort or wait" -v
```

Expected:

- PASS

**Step 5: Commit**

Run:

```bash
git add verl/experimental/agent_gateway/gateway.py verl/experimental/agent_gateway/types.py tests/experimental/agent_gateway/test_gateway_actor.py
git commit -m "feat: enforce gateway session lifecycle contracts"
```

### Task 2: Replace Text-Only Normalization with Canonical Structured Request Context

**Files:**
- Modify: `verl/experimental/agent_gateway/gateway.py`
- Modify: `verl/experimental/agent_gateway/types.py`
- Test: `tests/experimental/agent_gateway/test_gateway_actor.py`
- Check: `verl/utils/chat_template.py`
- Check: `verl/utils/dataset/rl_dataset.py`

**Step 1: Write the failing tests**

Add or update tests covering:

- malformed request vs unsupported-but-well-formed request are distinguished
- `tools` participates in request compatibility and is not silently ignored
- assistant `tool_calls` are retained in canonicalized request state
- tool message `tool_call_id` is retained
- multimodal `content` list is retained without flattening
- `name` is not silently dropped; PR1 either rejects or explicitly preserves it as reserved

Suggested test skeleton:

```python
@pytest.mark.asyncio
async def test_gateway_actor_rejects_unsupported_name_field(ray_runtime):
    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["X"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-name"))
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy", "messages": [{"role": "user", "name": "alice", "content": "hi"}]},
        )
    assert response.status_code in (400, 422)
```

**Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py -k "unsupported or malformed or tools or multimodal or tool_call or name" -v
```

Expected:

- failure because current implementation flattens messages and ignores structured fields

**Step 3: Implement canonical request handling**

Implement in `verl/experimental/agent_gateway/gateway.py`:

- replace `_normalize_messages()` text flattening with canonical structured request normalization
- track session truth source as canonical `messages + tools` request context
- enforce exact canonical prefix match on canonicalized request context
- enforce PR1 validation boundary:
  - required `messages`
  - no silent ignore of unsupported fields
  - stable handling of `tools`, `tool_calls`, `tool_call_id`, multimodal `content`
- keep `name` as a reserved decision point; do not silently erase it

Reuse VERL boundaries where reasonable:

- `verl/utils/chat_template.py`
- `verl/utils/dataset/rl_dataset.py`

Do not:

- introduce a new Gateway-private message DSL
- pull in `AgentLoop` orchestration state

**Step 4: Re-run the focused tests**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py -k "unsupported or malformed or tools or multimodal or tool_call or name" -v
```

Expected:

- PASS

**Step 5: Commit**

Run:

```bash
git add verl/experimental/agent_gateway/gateway.py verl/experimental/agent_gateway/types.py tests/experimental/agent_gateway/test_gateway_actor.py
git commit -m "feat: add canonical gateway request normalization"
```

### Task 3: Implement Tool Arguments Canonicalization and Prefix Continuation Regression Coverage

**Files:**
- Modify: `verl/experimental/agent_gateway/gateway.py`
- Test: `tests/experimental/agent_gateway/test_gateway_actor.py`

**Step 1: Write the failing tests**

Add or update tests covering:

- `tool_calls[*].function.arguments` accepts OpenAI-compatible string form when it parses to a JSON object
- non-JSON or non-object arguments are rejected
- semantically equivalent argument strings with different whitespace or key ordering do not cause false mismatch
- continuation within one trajectory preserves mixed `response_mask` values:
  - prompt-side continuation tokens are `0`
  - backend generation tokens are `1`

Suggested test skeleton:

```python
def test_gateway_actor_canonicalizes_tool_arguments_before_prefix_compare(...):
    ...
    assert trajectory.response_mask == [0, 0, 1, 1, ...]
```

**Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py -k "arguments or continuation or response_mask" -v
```

Expected:

- failure because current implementation has no structured tool-call comparison and weak continuation coverage

**Step 3: Implement the minimal canonicalization**

Implement in `verl/experimental/agent_gateway/gateway.py`:

- canonicalize `tool_calls[*].function.arguments`
- require JSON object after parsing
- compare prefix compatibility against stable canonical form
- keep continuation writes inside the active trajectory with correct `response_mask`

**Step 4: Re-run the focused tests**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py -k "arguments or continuation or response_mask" -v
```

Expected:

- PASS

**Step 5: Commit**

Run:

```bash
git add verl/experimental/agent_gateway/gateway.py tests/experimental/agent_gateway/test_gateway_actor.py
git commit -m "feat: stabilize gateway tool-call prefix matching"
```

### Task 4: Enforce Commit Point and No-Commit Failure Semantics

**Files:**
- Modify: `verl/experimental/agent_gateway/gateway.py`
- Test: `tests/experimental/agent_gateway/test_gateway_actor.py`

**Step 1: Write the failing tests**

Add or update tests covering:

- validation failure leaves session state unchanged
- backend generation failure leaves session state unchanged
- mismatch split is not durably committed before backend success
- `/complete` updates lifecycle and reward info only; it does not materialize trajectories

Suggested test skeleton:

```python
@pytest.mark.asyncio
async def test_gateway_actor_backend_failure_does_not_commit_partial_state(ray_runtime):
    ...
    state = ray.get(actor.get_session_state.remote("session-failure"))
    assert state["num_trajectories"] == 0
    assert state["has_active_trajectory"] is False
```

**Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py -k "failure or commit or no_commit or reward_info" -v
```

Expected:

- failure because current implementation mutates session state before backend success

**Step 3: Implement the commit-point fix**

Implement in `verl/experimental/agent_gateway/gateway.py`:

- treat prefix split, prompt-side delta, and active trajectory replacement as tentative state until backend success
- on backend failure or cancellation, restore or preserve the last committed session truth source
- ensure `/complete` does not materialize trajectories

**Step 4: Re-run the focused tests**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py -k "failure or commit or no_commit or reward_info" -v
```

Expected:

- PASS

**Step 5: Commit**

Run:

```bash
git add verl/experimental/agent_gateway/gateway.py tests/experimental/agent_gateway/test_gateway_actor.py
git commit -m "fix: make gateway chat mutations no-commit on failure"
```

### Task 5: Align GatewayManager and LLMServerManager with the Updated GatewayActor Contract

**Files:**
- Modify: `verl/experimental/agent_gateway/manager.py`
- Modify: `verl/experimental/agent_loop/agent_loop.py`
- Test: `tests/experimental/agent_gateway/test_gateway_manager.py`
- Test: `tests/experimental/agent_gateway/test_session_runtime.py`

**Step 1: Write the failing tests**

Add or update tests covering:

- `LLMServerManager` still owns Gateway lifecycle
- `GatewayManager` remains a routing/control-plane helper only
- session runtime methods preserve duplicate-session and terminal-state behavior through the manager layer
- backend failures and terminal lifecycle outcomes surface correctly through the manager/runtime layer

**Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_manager.py tests/experimental/agent_gateway/test_session_runtime.py -v
```

Expected:

- at least one failure if manager/runtime assumptions are out of sync with the updated GatewayActor contract

**Step 3: Implement the minimal integration updates**

Implement in `verl/experimental/agent_gateway/manager.py` and `verl/experimental/agent_loop/agent_loop.py`:

- keep `GatewayManager` internal to `LLMServerManager`
- preserve sticky `session_id -> gateway actor`
- forward updated lifecycle and waiting semantics correctly
- do not introduce a second framework-side control plane

**Step 4: Re-run the focused tests**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_manager.py tests/experimental/agent_gateway/test_session_runtime.py -v
```

Expected:

- PASS

**Step 5: Commit**

Run:

```bash
git add verl/experimental/agent_gateway/manager.py verl/experimental/agent_loop/agent_loop.py tests/experimental/agent_gateway/test_gateway_manager.py tests/experimental/agent_gateway/test_session_runtime.py
git commit -m "feat: align gateway runtime integration with session contracts"
```

### Task 6: Keep the Reference Framework Path Green Against the New Contracts

**Files:**
- Modify: `verl/experimental/agent_framework/openai_compatible_framework.py`
- Test: `tests/experimental/agent_framework/test_openai_compatible_framework.py`

**Step 1: Write the failing tests**

Add or update tests covering:

- the reference framework still creates sessions, drives chat completions, waits or finalizes, and assembles `DataProto`
- framework-side behavior remains valid when `GatewayActor` now enforces stricter lifecycle and request validation semantics

**Step 2: Run the focused tests and confirm failure if any**

Run:

```bash
pytest tests/experimental/agent_framework/test_openai_compatible_framework.py -v
```

Expected:

- PASS or targeted failures that reveal framework/runtime drift

**Step 3: Apply only the minimal compatibility fixes**

Implement:

- minimal adjustments required to keep the reference framework aligned with the new Gateway contracts
- no new orchestration layer
- no SWE or Retool expansion in this task

**Step 4: Re-run the focused tests**

Run:

```bash
pytest tests/experimental/agent_framework/test_openai_compatible_framework.py -v
```

Expected:

- PASS

**Step 5: Commit**

Run:

```bash
git add verl/experimental/agent_framework/openai_compatible_framework.py tests/experimental/agent_framework/test_openai_compatible_framework.py
git commit -m "test: keep openai-compatible framework aligned with gateway contracts"
```

### Task 7: Full Verification Pass

**Files:**
- Modify as needed from previous tasks
- Test: `tests/experimental/agent_gateway/test_gateway_actor.py`
- Test: `tests/experimental/agent_gateway/test_gateway_manager.py`
- Test: `tests/experimental/agent_gateway/test_session_runtime.py`
- Test: `tests/experimental/agent_framework/test_openai_compatible_framework.py`

**Step 1: Run the full focused verification suite**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py tests/experimental/agent_gateway/test_gateway_manager.py tests/experimental/agent_gateway/test_session_runtime.py tests/experimental/agent_framework/test_openai_compatible_framework.py -v
```

Expected:

- PASS

**Step 2: Run targeted grep review for outdated assumptions**

Run:

```bash
rg -n "encode_messages|messages only|silently ignore|duplicate create|completed_flag|aborted_flag" verl/experimental/agent_gateway verl/experimental/agent_framework tests/experimental/agent_gateway tests/experimental/agent_framework
```

Expected:

- no remaining code paths that clearly contradict the updated contracts

**Step 3: Update docs only if implementation exposed mismatches**

If the code revealed any small contract/doc wording drift, update:

- `cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-design-v2.md`
- `cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-implementation-plan-v2.md`

Do not expand scope beyond PR1.

**Step 4: Commit the final verification batch**

Run:

```bash
git add cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-design-v2.md cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-implementation-plan-v2.md verl/experimental/agent_gateway/gateway.py verl/experimental/agent_gateway/manager.py verl/experimental/agent_gateway/types.py verl/experimental/agent_loop/agent_loop.py verl/experimental/agent_framework/openai_compatible_framework.py tests/experimental/agent_gateway/test_gateway_actor.py tests/experimental/agent_gateway/test_gateway_manager.py tests/experimental/agent_gateway/test_session_runtime.py tests/experimental/agent_framework/test_openai_compatible_framework.py
git commit -m "feat: implement gateway actor pr1 session contracts"
```

## Deferred Follow-Up

- token ingress
- `AgentLoopManager` migration
- streaming
- full `name` semantics
- richer metadata propagation into `DataProto`
- exactly-once retry / idempotency keys
- SWE-Agent and Retool validation expansion

## Reviewer Checklist

- PR1 remains centered on `/v1/chat/completions`
- `AgentFramework` stays thin
- `GatewayManager` stays internal to `LLMServerManager`
- no text-only flattening remains in the Gateway truth source
- backend failure cannot partially commit session truth source
- unsupported fields are not silently ignored
