# AgentFramework & AgentGateway Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a staged AgentFramework + AgentGateway rollout path for VERL without breaking the existing AgentLoop path, with a ReactAgentLoop bridge and lightweight end-to-end reward-curve validation.

**Architecture:** We keep `AgentFrameworkManager` parallel to `AgentLoopManager`, target a future shared rollout runtime, introduce `AgentGatewayManager`/`GatewayActor` for trajectory capture, and assemble training samples through a dedicated `TrajectoryAssembler`. Xibin will likely land rollout-server management extraction upstream, so this plan does not include a standalone runtime-extraction PR; instead, `AgentFrameworkManager` temporarily keeps runtime creation manager-local behind a narrow seam that can be replaced once upstream extraction lands. Legacy compatibility is provided by `VerlLoopFramework`, while training-visible output must stay aligned with current `AgentLoopWorker` contracts.

**Tech Stack:** Ray actors, FastAPI/aiohttp-style async HTTP serving, Hydra/OmegaConf, DataProto/TensorDict, existing VERL rollout runtime and tokenizer/processor utilities.

---

## Series Layout

- Upstream-owned prerequisite, not part of this implementation series:
  - rollout server / runtime extraction
- This series PR1:
  - Gateway + Framework main loop
  - React bridge
  - component/equivalence tests
  - lightweight E2E with reward-curve evidence
- This series PR2+:
  - broader bridge coverage
  - Retool / SWE validation
  - deferred compatibility features

### Task 1: Add AgentGatewayManager and GatewayActor Core

**Files:**
- Create: `verl/experimental/agent_gateway/__init__.py`
- Create: `verl/experimental/agent_gateway/types.py`
- Create: `verl/experimental/agent_gateway/manager.py`
- Create: `verl/experimental/agent_gateway/gateway.py`
- Test: `tests/experimental/agent_gateway/test_gateway_manager.py`
- Test: `tests/experimental/agent_gateway/test_gateway_actor.py`

**Step 1: Write failing manager and actor lifecycle tests**

- Cover:
  - `create_session`
  - `finalize_session`
  - `abort_session`
  - `wait_for_completion`
  - sticky `session_id -> gateway actor`
  - least-active-session routing

**Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_manager.py tests/experimental/agent_gateway/test_gateway_actor.py -v
```

Expected:

- Missing module / missing actor API failures

**Step 3: Implement core gateway control plane and session state**

- Add `AgentGatewayManager` as a Ray actor
- Add `GatewayActor` with:
  - async multi-session handling
  - per-session lock
  - normalized message prefix checking
  - multiple trajectories per session

**Step 4: Run focused tests**

Run:

```bash
pytest tests/experimental/agent_gateway/test_gateway_manager.py tests/experimental/agent_gateway/test_gateway_actor.py -v
```

Expected:

- Lifecycle and routing tests pass

**Step 5: Commit on feature branch**

```bash
git add verl/experimental/agent_gateway tests/experimental/agent_gateway
git commit -m "feat: add agent gateway manager and actor core"
```

### Task 2: Add TrajectoryAssembler and Core AgentFramework Types

**Files:**
- Create: `verl/experimental/agent_framework/__init__.py`
- Create: `verl/experimental/agent_framework/framework.py`
- Create: `verl/experimental/agent_framework/assembler.py`
- Create: `verl/experimental/agent_framework/types.py`
- Test: `tests/experimental/agent_framework/test_assembler.py`

**Step 1: Write failing assembler tests**

- Cover:
  - prompt left padding
  - response right padding
  - `response_mask`
  - `input_ids`
  - `attention_mask`
  - `position_ids`
  - `rm_scores`
  - `__num_turns__`
  - optional `rollout_log_probs`
  - optional `routed_experts` when provided

**Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/experimental/agent_framework/test_assembler.py -v
```

Expected:

- Missing assembler or output contract failures

**Step 3: Implement minimal core framework and assembler**

- Add `Trajectory` and session result types
- Add `AgentFramework` abstract interface
- Add `TrajectoryAssembler` aligned to current training-visible `DataProto` contract
- Keep reward/teacher/legacy enrichers out of the core path

**Step 4: Run focused tests**

Run:

```bash
pytest tests/experimental/agent_framework/test_assembler.py -v
```

Expected:

- All output-contract tests pass

**Step 5: Commit on feature branch**

```bash
git add verl/experimental/agent_framework tests/experimental/agent_framework/test_assembler.py
git commit -m "feat: add agent framework core and trajectory assembler"
```

### Task 3: Add AgentFrameworkManager and FrameworkWorker

**Files:**
- Create: `verl/experimental/agent_framework/manager.py`
- Modify: `verl/experimental/agent_framework/framework.py`
- Test: `tests/experimental/agent_framework/test_manager_worker.py`

**Step 1: Write failing orchestration tests**

- Cover:
  - manager creates workers and gateway subsystem
  - manager chunks `DataProto`
  - worker runs per-sample session flow
  - manager aggregates worker outputs

**Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/experimental/agent_framework/test_manager_worker.py -v
```

Expected:

- Missing manager/worker flow failures

**Step 3: Implement manager and worker flow**

- `AgentFrameworkManager` owns:
  - framework workers
  - gateway subsystem
  - temporary manager-local runtime seam or injected runtime dependency
- `FrameworkWorker` owns:
  - framework instance
  - gateway manager client
  - per-chunk concurrent session execution

**Step 4: Run focused tests**

Run:

```bash
pytest tests/experimental/agent_framework/test_manager_worker.py -v
```

Expected:

- Orchestration tests pass

**Step 5: Commit on feature branch**

```bash
git add verl/experimental/agent_framework/manager.py tests/experimental/agent_framework/test_manager_worker.py
git commit -m "feat: add agent framework manager and worker orchestration"
```

Notes:

- The manager-local runtime seam in this task is intentionally transitional.
- Keep the seam narrow so upstream runtime extraction can replace it with minimal diff.
- Do not proactively refactor `AgentLoopManager` or `fully_async_policy` in this series unless required by test breakage.

### Task 4: Implement ReactAgentLoop Bridge via VerlLoopFramework

**Files:**
- Create: `verl/experimental/agent_framework/verl_loop_framework.py`
- Modify: `recipe/langgraph_agent/react_agent_loop.py`
- Modify: `recipe/langgraph_agent/chat_model.py`
- Test: `recipe/langgraph_agent/test_react_agent_loop.py`
- Test: `tests/experimental/agent_framework/test_react_bridge_equivalence.py`

**Step 1: Write failing bridge equivalence tests**

- Compare current AgentLoop path vs new Framework/Gateway path for:
  - `prompts`
  - `responses`
  - `response_mask`
  - `input_ids`
  - `attention_mask`
  - `position_ids`
  - `__num_turns__`
  - optional `rollout_log_probs`
  - optional `routed_experts`

**Step 2: Run tests to verify failure**

Run:

```bash
pytest recipe/langgraph_agent/test_react_agent_loop.py tests/experimental/agent_framework/test_react_bridge_equivalence.py -v
```

Expected:

- Missing bridge implementation or output mismatches

**Step 3: Implement `VerlLoopFramework` and React bridge path**

- Use Gateway as the trajectory truth source
- Reuse React agent-specific execution behavior
- Avoid keeping legacy trajectory bookkeeping as production truth

**Step 4: Run focused tests**

Run:

```bash
pytest recipe/langgraph_agent/test_react_agent_loop.py tests/experimental/agent_framework/test_react_bridge_equivalence.py -v
```

Expected:

- Bridge tests pass

**Step 5: Commit on feature branch**

```bash
git add verl/experimental/agent_framework/verl_loop_framework.py recipe/langgraph_agent/react_agent_loop.py recipe/langgraph_agent/chat_model.py recipe/langgraph_agent/test_react_agent_loop.py tests/experimental/agent_framework/test_react_bridge_equivalence.py
git commit -m "feat: bridge react agent loop through verl loop framework"
```

### Task 5: Add Optional Routed Experts Support

**Files:**
- Modify: `verl/experimental/agent_gateway/types.py`
- Modify: `verl/experimental/agent_gateway/gateway.py`
- Modify: `verl/experimental/agent_framework/assembler.py`
- Test: `tests/experimental/agent_framework/test_assembler.py`

**Step 1: Write failing optional capability test**

- Cover:
  - routed experts propagated from trajectory data into batch output
  - no regression when routed experts are absent

**Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/experimental/agent_framework/test_assembler.py -k routed_experts -v
```

Expected:

- Missing routed experts propagation

**Step 3: Implement optional enricher**

- Treat `routed_experts` as optional capability
- Align padding/placement behavior with current `AgentLoopWorker`

**Step 4: Run focused tests**

Run:

```bash
pytest tests/experimental/agent_framework/test_assembler.py -v
```

Expected:

- Assembler tests still pass with and without routed experts

**Step 5: Commit on feature branch**

```bash
git add verl/experimental/agent_gateway/types.py verl/experimental/agent_gateway/gateway.py verl/experimental/agent_framework/assembler.py tests/experimental/agent_framework/test_assembler.py
git commit -m "feat: support optional routed experts in agent framework assembler"
```

### Task 6: Add Lightweight End-to-End Acceptance Run for the First Feature PR

**Files:**
- Create: `tests/special_e2e/run_agent_framework_react_bridge.sh`
- Modify: relevant lightweight config or example files under `examples/` or `tests/special_e2e/`
- Document: `cxb_dev/docs/plans/2026-03-30-agent-framework-gateway-design.md`

**Step 1: Define the acceptance workload**

- Use a lightweight React/tool-use path
- Keep:
  - small model
  - small dataset
  - small training step count
- Require reward curve output and basic learnability signal

**Step 2: Run the acceptance script locally**

Run:

```bash
bash tests/special_e2e/run_agent_framework_react_bridge.sh
```

Expected:

- End-to-end rollout/training completes
- Reward curve artifact or logged metric trend is produced

**Step 3: Adjust config and implementation until stable**

- Fix config path issues
- Fix framework/gateway lifecycle bugs surfaced only in E2E
- Keep the workload minimal but real

**Step 4: Re-run acceptance**

Run:

```bash
bash tests/special_e2e/run_agent_framework_react_bridge.sh
```

Expected:

- Stable success
- Reward curve or equivalent logged signal available for PR evidence

**Step 5: Commit on feature branch**

```bash
git add tests/special_e2e/run_agent_framework_react_bridge.sh
git commit -m "test: add lightweight e2e validation for agent framework bridge"
```

### Task 7: Post-PR Validation Backlog (Do Not Block First Acceptance)

**Files:**
- Future work only; do not block the first feature PR

**Step 1: Track deferred validation**

- `Retool` validation
- `SWE-Agent` validation
- `fully_async_policy` support on AgentFramework path
- teacher logprobs / distillation path
- fuller multimodal and legacy compatibility

**Step 2: Convert backlog into follow-up PR scopes**

- Separate bridge-expansion work from core runtime work
- Keep acceptance criteria explicit

**Step 3: Document in PR description**

- State what is implemented now
- State what is intentionally deferred

**Step 4: Commit planning/docs changes if needed**

```bash
git add cxb_dev/docs/plans
git commit -m "docs: record deferred validation backlog for agent framework"
```
