# Pre-merge Global Review: Agent Framework + Gateway

## 目标

对 `feature/agent_framework` 分支准备合入 `main` 的全部代码变更做全局 review。Review 完成后将合为一个 commit 推到 main 并向主仓提 PR。

## 变更范围

查看相对 main 的完整 diff（排除 cxb_dev/ 目录）：
```bash
cd /home/cxb/rl_framework/verl && git diff main..HEAD -- verl/ tests/
```
加上当前未提交的修改：
```bash
git diff HEAD -- verl/ tests/
```

两者合并即为最终要提交的完整变更。涉及的代码文件：

**新增实现文件：**
- `verl/experimental/agent_framework/types.py` — Trajectory, TrajectoryRewardContext, RewardFn, SessionRuntime Protocol
- `verl/experimental/agent_framework/assembler.py` — TrajectoryAssembler（Trajectory → DataProto）
- `verl/experimental/agent_framework/helpers.py` — validate_trajectory, normalize_trajectory_rewards
- `verl/experimental/agent_framework/framework.py` — AgentFramework 基类 + OpenAICompatibleAgentFramework

**修改的实现文件：**
- `verl/experimental/agent_gateway/gateway.py` — _GatewayActor 大量重构
- `verl/experimental/agent_gateway/types.py` — GatewaySessionState 字段变更
- `verl/experimental/agent_loop/agent_loop.py` — AsyncLLMServerManager 合并 gateway runtime

**新增测试文件：**
- `tests/experimental/agent_framework/test_assembler.py`
- `tests/experimental/agent_framework/test_openai_compatible_framework.py`
- `tests/experimental/agent_framework/test_reward_helpers.py`
- `tests/experimental/agent_gateway/support.py` — 测试基础设施（FakeTokenizer, backends 等）
- `tests/experimental/agent_gateway/test_gateway_actor.py`
- `tests/experimental/agent_gateway/test_gateway_manager.py`
- `tests/experimental/agent_gateway/test_session_runtime.py`

## 参考文件

1. 设计文档：`/home/cxb/rl_framework/verl/cxb_dev/AGENTS.md`
2. VERL 多轮 tokenization 文档：`/home/cxb/rl_framework/verl/docs/sglang_multiturn/multiturn.rst`
3. VERL 现有 agent loop：`verl/experimental/agent_loop/tool_agent_loop.py`
4. VERL 现有 reward manager：`verl/experimental/reward_loop/reward_manager/naive.py`

## Review 要求

### 1. 行为逻辑对齐 VERL 现有实践

- 增量编码是否与 `ToolAgentLoop` 的 `remove_system_prompt` 模式一致？
- `RewardFn` 接口是否与 `NaiveRewardManager.run_single` 的语义一致？
- `TrajectoryAssembler` 的 padding/mask 逻辑是否与 `AgentLoopWorker._agent_loop_postprocess` 一致？
- `AsyncLLMServerManager` 的 gateway runtime 集成是否与现有 server manager 模式一致？
- gateway 的 prefix check（直接 dict equality）是否与 vLLM token-level prefix caching 的行为一致？

### 2. YAGNI 原则 — 避免过度设计

- 是否存在当前没有使用场景的抽象或参数？
- 校验代码和防御代码是否存在冗余？
- normalize 函数中的字段过滤是否过于激进或不必要？
- `SessionRuntime` Protocol 是否必要，还是可以直接依赖 `AsyncLLMServerManager`？

### 3. 测试精简

这是 PR 前的最后一轮，需要对测试用例进行精简：
- **保留**：验证核心行为的集成测试、适合 CI 流水线的测试
- **精简或删除**：纯 TDD 过程中产生的细粒度单元测试（如果行为已被更粗粒度的测试覆盖）
- **精简或删除**：过度防御的边界测试（如果对应的边界检查本身就不必要）
- 请具体指出哪些测试建议保留、哪些建议删除，并说明理由

### 4. 代码质量

- 是否有明显的 bug 或逻辑不一致？
- 是否有不必要的 TODO 可以在本轮关闭？
- 命名是否清晰一致？
- 是否有残留的死代码？

## 工作方式

- 以当前代码状态为准（HEAD + 未提交修改）
- 客观审视，不迎合已有决策
- 保持代码简洁是首要原则
- 输出格式：按文件分组，每个文件列出 findings，标注严重程度（Critical / Important / Suggestion）
- 最后给出一个总结：是否 ready to merge，以及 merge 前必须修复的 blocking issues（如果有）
