# AgentFramework Review And Next-Phase Handoff

**Purpose:** Provide a practical manual review entrypoint for the current `feature/agent_framework` branch and a clean handoff for the next planning window.

**Scope boundary:** This note is anchored to the current branch state after the recent `GatewayActor` contract tightening. It is not a new architecture baseline and does not replace `2026-04-01-agent-framework-gateway-design-v2.md` or `2026-04-01-agent-framework-gateway-implementation-plan-v2.md`.

---

## 1. Manual Review Entry Points

There is not yet a single official recipe or CLI entrypoint for the new AgentFramework + Gateway path. For manual review, the most reliable entrypoints are the test-backed paths below.

### A. GatewayActor contract review

This is the recommended first review pass because most of the recent design convergence landed here.

**Runnable entrypoint**

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py -v
```

**Recommended reading order**

1. `tests/experimental/agent_gateway/test_gateway_actor.py`
2. `verl/experimental/agent_gateway/gateway.py`
3. `verl/experimental/agent_gateway/types.py`
4. `tests/experimental/agent_gateway/support.py`

**What this pass should verify**

- session lifecycle contract: `ACTIVE -> COMPLETED -> FINALIZED / ABORTED`
- same-session serialization boundary
- duplicate session rejection
- malformed vs unsupported request handling
- canonical request context handling for `messages + tools`
- `tool_calls` / `tool_call_id` / multimodal content retention
- prefix continuation and mismatch split behavior
- backend failure no-commit semantics

### B. Session runtime ownership / serving integration review

This pass reviews whether the current `LLMServerManager`-owned runtime path is internally coherent.

**Runnable entrypoints**

```bash
pytest tests/experimental/agent_gateway/test_session_runtime.py -v
pytest tests/experimental/agent_gateway/test_gateway_manager.py -v
```

**Recommended reading order**

1. `tests/experimental/agent_gateway/test_session_runtime.py`
2. `verl/experimental/agent_loop/agent_loop.py`
   - focus on `LLMServerManager` / `AsyncLLMServerManager`
3. `verl/experimental/agent_gateway/manager.py`
4. `tests/experimental/agent_gateway/test_gateway_manager.py`
5. `verl/experimental/agent_gateway/gateway.py`

**What this pass should verify**

- gateway ownership and shutdown are local to `LLMServerManager`
- session-to-gateway sticky routing is stable
- remote call failure does not prematurely drop routing state
- framework/runtime-facing APIs are thin pass-throughs rather than a second orchestration layer

### C. Thin framework reference-path review

This pass checks whether the current reference framework is still thin and whether its validation value is correctly scoped.

**Runnable entrypoint**

```bash
pytest tests/experimental/agent_framework/test_openai_compatible_framework.py -v
```

**Recommended reading order**

1. `tests/experimental/agent_framework/test_openai_compatible_framework.py`
2. `verl/experimental/agent_framework/openai_compatible_framework.py`
3. `verl/experimental/agent_framework/assembler.py`
4. `verl/experimental/agent_framework/framework.py`
5. `verl/experimental/agent_gateway/manager.py`
6. `verl/experimental/agent_loop/agent_loop.py`

**What this pass should verify**

- top-level contract remains `generate_sequences(prompts: DataProto) -> DataProto`
- framework creates/finalizes sessions but does not absorb Gateway session logic
- current validation path is representative only for a minimal remote-style completion loop

### Suggested overall review order

If time is limited, use this exact order:

1. `test_gateway_actor.py`
2. `gateway.py`
3. `types.py`
4. `test_session_runtime.py`
5. `agent_loop.py` (`LLMServerManager` only)
6. `manager.py`
7. `test_openai_compatible_framework.py`
8. `openai_compatible_framework.py`
9. `assembler.py`

---

## 2. Next-Phase Planning Scope

The next planning window should stay in design / investigation mode first, not jump directly into implementation.

### A. Chat-template alignment

**Question to answer**

- Is PR2 alignment aiming for only schema-level compatibility, or real tokenization-level parity with VERL `apply_chat_template(...)`?

**Code to inspect**

1. `verl/experimental/agent_gateway/gateway.py`
2. `verl/experimental/agent_loop/agent_loop.py`
3. `verl/utils/chat_template.py`
4. `verl/experimental/agent_loop/utils.py`
5. `tests/experimental/agent_loop/test_basic_agent_loop.py`
6. `tests/experimental/agent_loop/test_multi_modal.py`

**Specific risks to assess**

- `GatewayActor` still uses `tokenizer.encode_messages(...)`, not `apply_chat_template(...)`
- multimodal processing in VERL depends on `processor` and `process_vision_info(...)`, not only structured message retention
- system prompt / generation prompt / model-specific template behavior may break incremental prefix assumptions
- some model paths already need special handling, e.g. `gpt-oss`

**Expected output**

- explicit recommendation on whether the next step is:
  - full tokenization-level alignment
  - a staged adapter layer
  - or an explicit defer with risk acceptance

### B. Completion return path completeness

The current PR1 work mostly tightened the request/tokenization boundary. The next window should inspect whether the response-side path is complete enough for tool-using agents.

**Question to answer**

- For completion-driven agents, is preserving request-side `tools` / `tool_calls` enough, or do we also need a canonical response-side parse path from generated tokens back into structured assistant/tool messages?

**Code to inspect**

1. `verl/experimental/agent_loop/tool_agent_loop.py`
2. `verl/experimental/agent_loop/tool_parser.py`
3. `tests/experimental/agent_loop/test_basic_agent_loop.py`
4. `tests/experimental/agent_loop/test_multi_modal.py`
5. `recipe/retool/README.md`
6. `recipe/retool/retool.py`

**Specific questions**

- where should tool-call parsing live if an agent runs through Gateway chat completions?
- should parsing stay entirely agent-side, or is some Gateway/runtime helper needed?
- do returned assistant messages need stronger structure guarantees than the current minimal `choices[0].message.content` path?
- do multimodal tool responses require additional runtime support beyond the request-side canonicalization already added in PR1?

### C. SWE-agent adaptation and validation

**Question to answer**

- Should SWE-agent be the primary post-PR1 validation path, and if yes, what must change for it to run on the completion path rather than its current proxy path?

**Code to inspect**

1. `recipe/swe_agent/README.md`
2. `recipe/swe_agent/swe_agent_loop.py`
3. `recipe/swe_agent/model_proxy.py`
4. `verl/experimental/agent_framework/openai_compatible_framework.py`
5. `verl/experimental/agent_gateway/gateway.py`

**Specific risks to assess**

- current SWE-agent flow already has its own model-proxy control plane
- moving it to Gateway completions may create overlap with existing proxy responsibilities
- chat-template parity and tool-call return-path gaps may block a faithful migration
- review value depends on whether SWE-agent exercises the exact contracts PR2 wants to stabilize

### D. Independent review for remaining fixups

This pass is meant to find issues that are not part of the already-settled design debate.

**Likely review targets**

- design-doc wording drift vs current implementation
  - `name` currently remains a reserved / unsupported field, so docs should not imply broad support
- long-lived actor bookkeeping tradeoffs
  - e.g. terminal-phase map growth
- other non-generation request fields that may still leak into backend sampling params
- metadata propagation gaps between session debug state, trajectories, and assembled `DataProto`
- missing representative tests on the framework/SWE side rather than only Gateway unit tests

---

## 3. Expected Deliverables From The Next Window

The next window should preferably produce:

1. A written design assessment for chat-template alignment, including explicit go / no-go recommendation.
2. A written assessment of response-side completeness for tool-using agents.
3. A validation recommendation comparing:
   - minimal mock completion path
   - SWE-agent completion-path adaptation
   - keeping both with different roles
4. A short list of additional concrete fixups discovered during independent review.

If the conclusions are stable, then write a new implementation/discovery plan rather than modifying the current PR1 implementation plan in place.

---

## 4. Suggested Handoff Prompt

Use the following prompt in a new window:

```text
请以 `feature/agent_framework` 分支当前状态为准继续工作，不要回退到旧分支设计。

先阅读以下文件：
1. /home/cxb/rl_framework/verl/cxb_dev/AGENTS.md
2. /home/cxb/rl_framework/verl/cxb_dev/docs/project-context.md
3. /home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-design-v2.md
4. /home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-01-agent-framework-gateway-implementation-plan-v2.md
5. /home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-04-03-agent-framework-review-handoff.md

然后重点做“下一阶段工作规划”，暂时不要直接改代码。请客观评估并输出书面结论，必要时可使用 subagents 做局部 review / debate。

本轮重点：

A. chat-template 对齐
- 评估 Gateway 从当前 `encode_messages(...)` 路径，对齐到 VERL `apply_chat_template(...)` / `initialize_system_prompt(...)` / processor / multimodal helper 的可行性、主要阻碍和风险。
- 需要明确区分：
  - schema-level alignment
  - tokenization-level parity
- 请判断下一步应该：
  1. 直接推进完整对齐
  2. 分阶段做 adapter
  3. 明确 defer，并说明 maintainer 需要接受什么风险

B. completion 返回路径是否完备
- 当前我们重点收束了 request / truth-source / prefix / failure semantics。
- 但请继续评估从“模型输出 token -> assistant structured message / tool call / tool response continuation”这一侧是否完备。
- 请重点看：
  - `/home/cxb/rl_framework/verl/verl/experimental/agent_loop/tool_agent_loop.py`
  - `/home/cxb/rl_framework/verl/verl/experimental/agent_loop/tool_parser.py`
  - `/home/cxb/rl_framework/verl/recipe/retool/README.md`
  - `/home/cxb/rl_framework/verl/recipe/retool/retool.py`
- 需要回答：
  - tool parser 是否应保持 agent-side responsibility
  - Gateway/runtime 是否需要补一个 response-side helper contract
  - 当前 completion path 是否足以支撑 tool-using agents 的真实 validation

C. SWE-agent 改造与验证
- 请评估 SWE-agent 从当前 model-proxy path 改为走 completion path 的意义、代表性、成本和主要阻碍。
- 重点阅读：
  - `/home/cxb/rl_framework/verl/recipe/swe_agent/README.md`
  - `/home/cxb/rl_framework/verl/recipe/swe_agent/swe_agent_loop.py`
  - `/home/cxb/rl_framework/verl/recipe/swe_agent/model_proxy.py`
  - `/home/cxb/rl_framework/verl/verl/experimental/agent_framework/openai_compatible_framework.py`
- 需要给出：
  - 是否值得作为下一阶段主 validation path
  - 如果值得，最小改造路径是什么
  - 如果不值得，更好的 validation 方案是什么

D. 独立 review 继续查漏补缺
- 目标不是重复此前 GatewayActor 设计讨论，而是找出剩余不一致、隐藏风险和应补测试点。
- 请明确区分：
  - 当前代码事实
  - 已冻结决策
  - 你的推断 / 建议

输出要求：
1. Executive summary
2. Chat-template alignment assessment
3. Completion return-path assessment
4. SWE-agent validation assessment
5. Additional review findings
6. Recommended next planning artifact

如果你认为需要写新文档，请先说明应新增什么文档以及为什么，不要直接进入大规模实现。
```
