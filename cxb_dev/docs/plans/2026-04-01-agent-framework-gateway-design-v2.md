# AgentFramework 与 AgentGateway 设计文档 v2

> 本文档是 2026-04-01 的设计收敛版本，覆盖并取代 [2026-03-30-agent-framework-gateway-design.md](/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-03-30-agent-framework-gateway-design.md) 作为当前执行基线。旧文档保留为讨论历史记录，不再作为当前实现依据。

## 1. 背景与目标

本文档基于：

- RFC [agentFramework_agentgateway_rfc.md](/home/cxb/MATE-reboot/docs/rfc/agentFramework_agentgateway_rfc/agentFramework_agentgateway_rfc.md)
- VERL `main` 分支现状
- 与 Xibin 的最新对齐结果
- 对现有 `AgentLoop`、Aliyun Remote Agent、AWS Bedrock / AgentCore 目标场景的复盘

本版设计的核心目标是：

- 为 VERL 引入一个足够薄的 `AgentFramework` 抽象，使其能统一承接 VERL-native agent、remote agent、hosted agent framework。
- 为 VERL 引入一个由 `LLMServerManager` 拥有的 `AgentGateway` 子系统，通过 OpenAI-compatible `/v1/chat/completions` 截获请求并组装 trajectories。
- 明确首版主路径是 chat-completions / session-based integration，而不是现有 `AgentLoopManager` 的立即迁移。
- 为后续 `AgentLoopManager` 迁移预留 token request ingress，但不把它作为首版阻塞项。

非目标：

- 首版不迁移现有 `AgentLoopManager` 到新 Gateway 主路径。
- 首版不要求实现 token request ingress。
- 首版不要求支持全部 legacy `extra_fields`、teacher logprobs、fully async integration。
- 首版不要求把 Gateway 设计成独立于 serving runtime 的顶级平台服务。

## 2. 已收敛的核心决策

### 2.1 `AgentFramework` 是薄抽象

- `AgentFramework` 是统一抽象基类。
- 首版公共抽象接口只保留：
  - `generate_sequences(prompts: DataProto) -> DataProto`
- `run_session`
- `compute_reward`
- `finalize_session`
  这些都不进入 `AgentFramework` 的正式公共接口。

原因：

- trainer 真正依赖的是 `generate_sequences`。
- Aliyun / Bedrock / AgentCore 这类 remote/hosted framework 不适合被要求实现 VERL-native 的内部 hook。
- `compute_reward` 是 shared concern，但不是必须的 shared abstract method。

### 2.2 `AgentLoopManager` 是 `AgentFramework` 的一种特化实现

- `AgentLoopManager` 在长期方向上应当成为 `AgentFramework` 的一个 concrete implementation。
- 但 maintainer 已明确：`AgentLoopManager` 不作为首版迁移目标。
- 首版不再引入 `AgentFrameworkManager -> FrameworkWorker -> AgentFramework` 这套公共三层抽象。

含义：

- `FrameworkWorker` 不再是新架构的一层。
- `AgentLoopWorker` 若继续存在，也只能是 `AgentLoopManager` 的内部实现细节。

### 2.3 Gateway ownership 归 `LLMServerManager`

- Gateway 不由 `AgentFramework` 或 `AgentLoopManager` 拥有。
- Gateway 由 `LLMServerManager` 拥有并托管。
- `GatewayManager` 可以保留，但其定位是 `LLMServerManager` 内部的 session-routing / control-plane 子组件。
- `GatewayActor` 由 `LLMServerManager` 或其 runtime factory 创建，而不是由 `GatewayManager` 自己创建。

### 2.4 Framework 侧依赖的是窄 session capability

- Framework 不应长期直接绑定 `GatewayManager` 具体类型。
- Framework 应依赖一个由 `LLMServerManager` 暴露的窄 session capability。
- 首版建议能力集合：
  - `create_session`
  - `finalize_session`
  - `abort_session`
  - `wait_for_completion`

建议 session handle：

```python
@dataclass
class SessionHandle:
    session_id: str
    base_url: str | None = None
```

### 2.5 Gateway 首版只优先支持 `/v1/chat/completions`

- 首版正式主路径是 session-based OpenAI-compatible chat completion。
- token request ingress 需要预留扩展点，但不是首版优先实现项。
- `AgentLoopManager` 的迁移留到后续阶段。

这意味着：

- 首版优先服务 remote/hosted/OpenAI-compatible agent integration。
- 不为兼容现有 `AgentLoop` 立即把 Gateway 做成双协议入口。

### 2.6 token request ingress 是明确的后续扩展点

- 现有 `AgentLoop` 深度依赖 `AsyncLLMServerManager.generate(...)`。
- 如果未来要把 `AgentLoopManager` 迁移到 Gateway 主路径，需要一个可接收 token request 的 ingress。
- 该能力在本版设计中只做边界预留，不纳入首版交付。

### 2.7 Gateway 与 rollout server 是 `N:M` 关系

- `gateway_count` 可以独立配置。
- `gateway_count = 0` 表示不启动 Gateway 子系统。
- 这意味着 Gateway 与 rollout server 不是强 1:1 绑定关系。
- 更合理的拓扑是：
  - 数量独立
  - lifecycle 从属于 `LLMServerManager`
  - 运行时按路由关联

### 2.8 `wait_for_completion` 与 `/complete` 是一等能力

- `wait_for_completion` 不应删除。
- `POST /sessions/{id}/complete` 保留为可选完成信号与可选 `reward_info` 上传通道。
- 这对 remote/hosted agent 很重要。
- 对 subprocess / coroutine agent，它不是必经路径。

### 2.9 reward 是 shared concern，但不进入 framework 抽象接口

- trajectory reward assignment 是共性问题。
- 但 `compute_reward(...)` 不要求成为 `AgentFramework` 的抽象方法。
- 更合理的落位是：
  - concrete framework implementation 内部逻辑
  - 或 helper utility

首版建议沉淀的共享逻辑：

- reward normalization
- `trajectory -> DataProto` assembler
- 基本 validation

### 2.10 现有 `AgentLoop` 不应与 Gateway 主线双重记账

- 如果某条路径启用了 Gateway 并以其为 trajectory 真相源，则不能继续让旧 `AgentLoop` trajectory bookkeeping 作为生产真相源。
- 双重 bookkeeping 只适合迁移验证，不进入正式运行时设计。
- 因此现有 `AgentLoopManager` 迁移应当在 token ingress 方案明确后单独推进。

## 3. 三类 agent 场景分析

### 3.1 现有 `AgentLoop`

本地代码证据表明：

- trainer-facing contract 已经是 `generate_sequences`。
- 现有路径是 `AgentLoopManager -> AgentLoopWorker -> AgentLoopBase`。
- agent 多数直接调用 `AsyncLLMServerManager.generate(...)`。
- `AgentLoopManager` 当前还自己拥有 rollout server init、load balancer、worker placement、batch orchestration。

对新框架的要求：

- `AgentFramework` 薄接口是可接受的。
- 最大问题不是 framework 顶层接口，而是后续如何让 legacy native loop 过渡到 Gateway truth source。
- 现有 `AgentLoop` 迁移不应阻塞首版。

### 3.2 Aliyun Remote Agent

基于 RFC 和 issue 语义推断：

- 更像 remote service / hosted framework，而不是 VERL-native loop。
- 更关心：
  - `create_session`
  - `wait_for_completion`
  - `/complete`
  - `finalize_session`
- 对 `AgentFramework` 顶层抽象方法没有额外要求。

设计含义：

- 当前“薄 framework + 强 session runtime”方向更适合它。

### 3.3 AWS Bedrock / AgentCore

基于 RFC 和社区引用推断：

- 更像 hosted remote framework。
- 与 Aliyun 类似，更依赖稳定的 session/gateway capability。
- 会强化如下方向：
  - `AgentFramework` 只保留 `generate_sequences`
  - `wait_for_completion` 为正式能力
  - `/complete` 与 `reward_info` 上传通道保留

## 4. 模块落位

建议新增模块：

- `verl/experimental/agent_framework/`
  - `framework.py`
    - `AgentFramework`
  - `types.py`
    - `SessionHandle`
    - `Trajectory`
    - 相关轻量类型
  - `assembler.py`
    - `TrajectoryAssembler`
  - `helpers.py`
    - reward normalization / validation helpers

- `verl/experimental/agent_gateway/`
  - `types.py`
  - `manager.py`
    - `GatewayManager`，作为 `LLMServerManager` 的内部子组件
  - `gateway.py`
    - `GatewayActor`

现有模块处理：

- `verl/experimental/agent_loop/agent_loop.py`
  - 首版不做主路径迁移
  - 后续迁移时再处理 token ingress 与 legacy bookkeeping

- `verl/experimental/fully_async_policy/`
  - 首版不适配
  - 保持现状

## 5. 关键组件职责

### 5.1 `AgentFramework`

负责：

- 暴露统一 trainer-facing `generate_sequences`

不负责：

- 规定内部是 session-based 还是 token-based
- 规定 reward hook 形式
- 规定 worker 拓扑
- 规定 Gateway lifecycle

### 5.2 `LLMServerManager`

负责：

- rollout servers / backend handles ownership
- load balancer ownership
- Gateway 子系统 ownership
- 向 framework 暴露窄 session capability
- 维持 serving/runtime control-plane 能力

### 5.3 `GatewayManager`

定位：

- `LLMServerManager` 内部 session-routing 子组件

负责：

- `session_id -> gateway actor` sticky routing
- `create/finalize/abort/wait` forwarding

不负责：

- 创建 `GatewayActor`
- ownership 决策
- framework-facing 顶级抽象

### 5.4 `GatewayActor`

首版负责：

- session state
- `/v1/chat/completions`
- message-level prefix consistency
- trajectory assembly
- `/complete`

首版不负责：

- token ingress
- `AgentLoop` compatibility bridge

#### 5.4.1 session truth source

`GatewayActor` 内部对单个 session 的真相源拆分为三层：

- `message_history`
  - 当前已提交的消息级对话历史，用于 prefix consistency 判断
- `active_trajectory`
  - 当前仍在追加中的 trajectory buffer
- `trajectories`
  - 已 materialize 的稳定 trajectories

设计要求：

- 一个 turn 只有在 `GatewayActor` 完成 backend 返回处理并更新 `message_history` / `active_trajectory` 后，才算对该 session 提交成功。
- `finalize_session()` 只能返回已经 materialize 的 trajectories；如果存在 `active_trajectory`，必须先 materialize 再返回。
- 不引入额外的 segment/sub-trajectory 概念；prefix mismatch 直接切新 trajectory。

#### 5.4.2 session lifecycle

PR1 采用最小化 session phase model，而不是重型 workflow state machine。

最小状态：

- `ACTIVE`
  - session 已创建，可继续接收 chat completions
- `COMPLETED`
  - 已收到显式完成信号，不再接收新的 chat completions，但仍可 finalize
- `FINALIZED`
  - session 已完成 materialization 并从 runtime 中移除
- `ABORTED`
  - session 已被放弃并从 runtime 中移除

允许的主要转移：

- `create_session`: 创建 `ACTIVE` session
- `complete_session`: `ACTIVE -> COMPLETED`
- `finalize_session`: `ACTIVE|COMPLETED -> FINALIZED`
- `abort_session`: `ACTIVE|COMPLETED -> ABORTED`

约束：

- `FINALIZED` / `ABORTED` 是 terminal states。
- PR1 不要求 `/complete` 成为所有本地 path 的必经步骤，因此 `finalize_session()` 允许直接从 `ACTIVE` 进入 `FINALIZED`。
- 默认 `session_id` 应由 runtime/framework 侧生成唯一值，而不是默认复用外部业务 `uid`；PR1 推荐使用 `uuid4().hex` 这一类本地唯一 id 生成方式。
- 若调用方显式传入 `session_id`，其唯一性与幂等边界由调用方负责；Gateway 侧 duplicate reject 属于防御性校验，而不是主去重机制。
- duplicate `create_session(session_id)` 必须显式 reject，不能静默覆盖已有 session state。

#### 5.4.3 per-session serialization boundary

Frozen decision 中“same-session requests should be serialized”在实现上不应只覆盖 `/v1/chat/completions`，而应覆盖所有会修改 session truth source 的同-session 操作。

PR1 约束：

- 下列 mutating operations 必须共享同一个 per-session serialization boundary：
  - `chat_completions`
  - `complete_session`
  - `finalize_session`
  - `abort_session`
- `wait_for_completion()` 不应长时间持有该锁，但其完成条件只能在持锁的合法状态转移后触发。
- 目标不是防止“两个 finalize 同时到来”这种低频场景，而是防止 in-flight `chat_completions` 与 `complete/finalize/abort` 并发竞争同一个 session truth source。

#### 5.4.4 `/v1/chat/completions` request handling

PR1 的 `GatewayActor` 不是完整 OpenAI API reimplementation，而是一个 session-aware、OpenAI-compatible 的最小子集。

请求处理要求：

- 路径必须绑定到已存在 session；unknown session 返回明确 4xx，而不是隐式创建 session。
- `messages` 必须存在且非空；malformed request 返回明确 4xx。
- `GatewayActor` 内部保存的 session truth source 应是“可无损送入 VERL chat template 与 vision helper”的 canonical structured request context，而不是 text-only normalize 结果。
- PR1 的 prefix consistency 基于 canonical structured request context，而不是 token-level fuzzy match。
- 若请求消息是当前 `message_history` 的 prefix continuation：
  - 继续追加到同一个 `active_trajectory`
  - 新增的 prompt-side token 写入 `response_ids`
  - 这些 token 的 `response_mask` 置为 `0`
- 若请求消息与当前 `message_history` 不再 prefix-compatible：
  - 先 materialize 当前 `active_trajectory`
  - 再为新消息开启新的 trajectory

OpenAI compatibility boundary：

- PR1 正式主路径是 `messages` 驱动的 non-streaming chat completions。
- canonical session truth source 至少应覆盖：
  - `messages`
  - `tools`
- canonical message schema 应尽量贴近 OpenAI message shape，而不是引入新的 Gateway 私有消息协议；最小保留字段包括：
  - `role`
  - `content`
  - `tool_calls`
  - `tool_call_id`
  - `name`
- 为与 VERL 现有 chat-template / tool-agent 路径对齐，PR1 的设计基线应包含：
  - 顶层 `tools`
  - assistant message 上的 `tool_calls`
  - `role == "tool"` 的消息
  - multimodal `content` parts
- `content` 若是 multimodal parts，应保留结构化 list 及其顺序，不得为了 prefix matching 或内部存储而压平成纯文本。
- `tool_calls` 至少应保留 `id`、`type`、`function.name`、`function.arguments`。
- PR1 对 `tool_calls[*].function.arguments` 的最小 contract：
  - 外部请求可接受 OpenAI-compatible string form
  - canonicalization 时必须成功解析为 JSON object
  - internal comparison / prefix matching 必须基于稳定 canonical 表示，而不是原始字符串表示
  - 若解析失败，或结果不是 JSON object，不得静默接受为“看起来像 tool call”的历史
- 同一 active trajectory 内，`tools` 属于 request compatibility boundary 的一部分；若 `tools` 与已有 session truth source 不再 canonical-compatible，则不得静默沿用旧 trajectory。
- `GatewayActor` 不应把上述结构化字段静默压平为纯 `role/content` 文本，否则会破坏 chat template correctness 与后续 trajectory truth source。
- `name` 不应自动视为已支持字段；只有在某一类 message / model path 的语义被单独定义清楚后，才能宣称正式支持。
- PR1 不应通过静默忽略字段来制造“看似兼容”的假象。
- PR1 的 prefix match 采用 exact canonical prefix match：
  - 比较对象是 canonicalized `tools + messages`
  - 不做 token-level fuzzy matching
  - 不做基于文本近似的弱匹配
- 错误类型需要区分：
  - malformed request
    - 指请求缺少必需字段、字段类型不合法、结构不满足最小 chat-completions contract
  - unsupported-but-well-formed request
    - 指请求结构合法，但使用了 PR1 尚未支持的 capability，例如未定义语义的 `name`、streaming 或其它未承诺字段
- `GatewayActor` 应对这两类错误返回可区分的明确 4xx，而不是统一混成“malformed request”。

#### 5.4.5 `complete / wait / finalize / abort`

`POST /sessions/{id}/complete`：

- 指 agent-side completion signal，而不是 framework-facing public API。
- 是可选完成信号与可选 `reward_info` 上传通道。
- 对 remote/hosted agents 很重要；对本地 path 不是必经步骤。
- 成功处理后将 session 从 `ACTIVE` 转为 `COMPLETED`。
- `COMPLETED` 是 bridge state：表示 agent side 已声明完成，但 trajectory 尚未被 framework/runtime 侧消费并移除。
- 在 `ACTIVE` 上调用时成功。
- 在 `COMPLETED` 上重复调用时应视为幂等成功。
- 在 `FINALIZED` / `ABORTED` 上调用时应明确失败。

`wait_for_completion(session_id)`：

- 是 framework/runtime-facing synchronization primitive，不负责 finalize。
- 语义应当是“等待 session 到达可安全 finalize 的完成状态”，而不只是等待某个 event 被任意设置。
- 成功条件：
  - session 进入 `COMPLETED`
  - 或 session 已经是 `FINALIZED`
- 失败条件：
  - session 在等待期间进入 `ABORTED`
  - 或等待超时
- `ACTIVE` 状态下继续等待，不得把“仍可接受新 chat”误判为完成。
- 若 session 在等待期间被 abort，应以明确的 terminal failure 结束等待，而不是无限阻塞。

`finalize_session(session_id)`：

- 负责 materialize 任何尚未提交的 `active_trajectory`
- 复制当前 session 级别的 reward/session metadata 到返回的 trajectories（PR1 仅需最小调试与 reward 语义，不要求定义完整训练侧 metadata propagation 契约）
- 返回 trajectories 后，将 session 置为 `FINALIZED` 并从 runtime 中移除
- 是 framework/runtime 侧唯一的 consume-and-close 出口。
- 在 `ACTIVE` 上调用时允许直接 finalize；这对应本地 path 中 `/complete` 不是必经步骤的情况。
- 在 `COMPLETED` 上调用时允许 finalize。
- 在 `FINALIZED` / `ABORTED` 上调用时应明确失败，而不是伪幂等地重复返回旧结果。

`abort_session(session_id)`：

- 将 session 置为 `ABORTED` 并从 runtime 中移除
- 不产生训练可见 trajectories
- 不允许与 in-flight chat mutation 并发打架；必须经过同一 serialization boundary
- 在 `ACTIVE` / `COMPLETED` 上调用时成功。
- 在 `ABORTED` 上重复调用可视为幂等成功，用于清理路径。
- 在 `FINALIZED` 上调用时应明确失败。

#### 5.4.6 session debug metadata

PR1 允许 `GatewayActor` 维护最小 session debug metadata，例如：

- `metadata`
- `created_at` / `updated_at`
- completion / abort flags
- `num_trajectories`

但这类 metadata 默认只构成 runtime/debug contract，不自动等价于训练侧 `DataProto` contract。

#### 5.4.7 failure handling / retry semantics

PR1 需要明确定义 `GatewayActor` 的 commit point 和失败后的 no-op / no-commit 语义，避免 session truth source 被部分写坏。

最小约束：

- 对单次 chat completion，请求的 commit point 发生在：
  - backend generation 成功返回
  - 且 `GatewayActor` 已完成本轮 response processing，并准备好一致地更新 `message_history` / `active_trajectory`
- 在 commit point 之前，本轮请求中的 prefix split、prompt-side delta 追加、active trajectory replacement 都只能视为 tentative state；若本轮请求失败，必须整体丢弃，而不是部分写回 session。
- malformed request、unsupported-but-well-formed request、unknown session 等前置失败必须是纯 no-op：
  - 不更新 `message_history`
  - 不更新 `active_trajectory`
  - 不 materialize trajectory
  - 不改变 completion / abort state
- backend.generate(...) 失败、超时或被取消时，本轮 chat request 必须按 no-commit 处理：
  - 不提交本轮新增的 prompt-side delta
  - 不提交本轮 assistant response
  - 不因为本轮请求而 materialize 新 trajectory
  - session 应保持在本轮请求开始前的最后一个稳定状态
- `POST /complete` 只改变 lifecycle / optional reward info，不负责 trajectory materialization。
- PR1 不承诺跨网络故障场景下的 exactly-once completion semantics；若未来需要安全 retry / dedup，应通过显式 request id / idempotency key 机制单独设计。

#### 5.4.8 明确不做的事情

为避免过度设计，PR1 的 `GatewayActor` 不引入：

- backend generation 子状态
- token ingress 相关状态
- persistence / recovery state machine
- framework-level public lifecycle 扩张
- token-level fuzzy prefix matching

### 5.5 `TrajectoryAssembler` / helpers

负责：

- `trajectory -> DataProto`
- reward normalization
- validation

目标是对齐现有训练可观察输出契约：

- `prompts`
- `responses`
- `response_mask`
- `input_ids`
- `attention_mask`
- `position_ids`
- 可选 `rollout_log_probs`
- `rm_scores`
- `__num_turns__`
- 条件性 `routed_experts`

## 6. 首版接口建议

### 6.1 `AgentFramework`

```python
class AgentFramework(ABC):
    @abstractmethod
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        ...
```

### 6.2 session capability

```python
class SessionRuntime(Protocol):
    async def create_session(self, session_id: str, **kwargs) -> SessionHandle: ...
    async def finalize_session(self, session_id: str) -> list[Trajectory]: ...
    async def abort_session(self, session_id: str) -> None: ...
    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None: ...
```

### 6.3 token ingress 预留

不在首版正式实现，但设计上保留后续能力位：

- 接收 token request
- 服务于 `AgentLoopManager` 迁移
- 与 chat-completions 共享 session / trajectory model

## 7. 首版交付范围

### 本系列 PR1：Gateway + 薄 Framework 主闭环

交付：

- `AgentFramework` 薄抽象
- `LLMServerManager` 拥有 Gateway 子系统的边界设计
- `GatewayManager` / `GatewayActor`
- `/v1/chat/completions`
- `create/finalize/abort/wait`
- `/complete`
- `TrajectoryAssembler`
- 一个最小的 OpenAI-compatible / remote-style reference path
- 组件测试
- 一次轻量 E2E / inspection run

不交付：

- `AgentLoopManager` 迁移
- token ingress
- `ReactAgentLoop` bridge
- `Retool` / `SWE-Agent` validation

### 后续 PR

- token ingress
- `AgentLoopManager` 迁移
- `ReactAgentLoop` / `Retool` / `SWE-Agent` validation
- teacher logprobs / legacy compatibility / fully async 兼容

## 8. 风险与约束

主要风险：

- 如果首版仍试图同时兼容现有 `AgentLoop` 直连 `generate` 语义，容易出现 Gateway 与 legacy bookkeeping 的双重真相源冲突。
- 如果 framework 直接长期依赖 `GatewayManager` 具体类型，后续 `LLMServerManager` 内部调整空间会变小。
- 如果把 token ingress 与 chat-completions 同时作为首版主路径，复杂度会明显上升。

主要约束：

- maintainer 已明确首版优先 `/v1/chat/completions`。
- `AgentLoopManager` 后续再迁移。
- rollout runtime 抽取仍以上游后续工作为准。

## 9. 当前实施原则

- 先把 chat-completions 主路径做通。
- 抽象保持薄，不把 `AgentLoop` 的内部执行模型推成通用框架。
- Gateway ownership 明确归 `LLMServerManager`。
- Aliyun / Bedrock / AgentCore 这类 remote/hosted agent 视为首版设计的重要目标场景。
- `AgentLoop` 是后续迁移对象，不是首版阻塞项。
