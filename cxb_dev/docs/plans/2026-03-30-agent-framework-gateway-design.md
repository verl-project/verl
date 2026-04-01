# AgentFramework 与 AgentGateway 设计文档

## 1. 背景与目标

本文档基于 RFC `docs/rfc/agentFramework_agentgateway_rfc/agentFramework_agentgateway_rfc.md`、VERL `main` 分支现状，以及与用户的逐项讨论结果，收敛 AgentFramework / AgentGateway 在 VERL 中的首版设计。

目标：

- 为 VERL 引入独立的 `AgentFramework` 抽象，用于承接 agent session 生命周期、reward 计算与训练侧样本组装。
- 为 VERL 引入独立的 `AgentGateway` 抽象，用于通过 OpenAI-compatible API 截获 agent LLM 请求并组装 token-level trajectories。
- 保持现有 `AgentLoop` 体系可继续工作，并提供 `VerlLoopFramework` 作为桥接层。
- 采用渐进迁移策略，优先保证边界清晰、验证充分、上游可审。

非目标：

- 首版不要求一次迁移所有 `AgentLoopBase` 子类。
- 首版不要求完整覆盖全部 legacy capability。
- 首版不把 Gateway 设计成全局独立平台服务，而是作为 `AgentFrameworkManager` 组装和托管的子系统。

## 2. 已收敛的核心决策

### 2.1 管理器与运行时边界

- `AgentFrameworkManager` 与 `AgentLoopManager` 并列，不继承。
- 长期目标上，两者都应消费一个独立的共享 rollout runtime。
- 该共享 runtime 负责：
  - rollout replica / server 初始化
  - `GlobalRequestLoadBalancer`
  - `AsyncLLMServerManager`
  - runtime lifecycle (`clear_kv_cache` / `start_profile` / `stop_profile`)
- 该共享 runtime 不应长期内嵌在某一个 manager 里。

最终目标：

- `InferenceRuntime` 是 trainer-level shared infrastructure。
- `AgentLoopManager` 与 `AgentFrameworkManager` 都应以“依赖注入”的方式消费它。

开发期过渡方案：

- Xibin 已确认 rollout server 管理抽取近期会由上游推进，因此本工作流不自带独立 runtime extraction PR。
- 在上游抽取落地前，`AgentFrameworkManager.create()` 可以暂时以 manager-local 方式创建 runtime。
- 但 runtime 构造必须始终封装在明确可替换的 seam 后面，避免初始化逻辑散落。
- 当前实现应视为临时过渡层，而不是新的长期 ownership 方案。
- 一旦上游共享 runtime 落地，应优先替换到上游接口，而不是长期保留 manager-local ownership。

### 2.2 新架构的层级划分

扩展抽象层级：

- `AgentFrameworkManager -> AgentFramework -> GatewaySession`

运行时执行层级：

- `AgentFrameworkManager -> FrameworkWorker -> AgentFramework`

说明：

- `FrameworkWorker` 不可省略。它仍然是 actor 级共享执行容器。
- `session` 是工作单元，不是执行层。

### 2.3 Gateway 子系统拓扑

- `AgentFrameworkManager` 持有所有 `FrameworkWorker`。
- `AgentFrameworkManager` 同时创建并持有 Gateway 子系统。
- `AgentGatewayManager` 作为独立组件存在，但在 lifecycle / ownership 上从属于 `AgentFrameworkManager`。
- `AgentGatewayManager` 推荐实现为轻量 Ray actor，作为跨 worker 共享的 session sticky routing 真相源。

### 2.4 Gateway 路由与并发语义

- Gateway 首版负载策略：
  - `create_session` 时按 `least active sessions` 分配
  - session 创建后 sticky 到单个 `GatewayActor`
  - 不做运行中迁移
- `GatewayActor` 内部并发模型：
  - 跨 session 异步并发
  - 单 session 通过 `asyncio.Lock` 串行
  - 首版不引入显式请求队列
  - 同 session 重入请求等待，不返回 `409/429`
- `GatewayManager` 不参与高频 `/chat/completions` 数据路径，只参与 session 生命周期控制面调用。

### 2.5 Prefix consistency 与 trajectory 术语

- 首版只做消息级 prefix consistency 检查。
- 不做 token-level fuzzy matching。
- prefix mismatch 时直接关闭当前 trajectory，开启新的 trajectory。
- 一个 `session` 可包含多个 `trajectories`。
- 不引入额外 `trajectory segment` 术语。

### 2.6 Session completion 与 reward 信息

- `POST /sessions/{id}/complete` 是可选完成信号与可选 `reward_info` 上传通道。
- 它不是所有 agent 的必经路径。
- 对 subprocess / coroutine agent，可通过进程退出或函数返回判断完成。
- 对 remote agent，必要时可依赖 `wait_for_completion()`。

### 2.7 首版 bridge 范围

- `VerlLoopFramework` 作为 compatibility bridge。
- 它只桥接 agent-specific 执行逻辑，不复用旧 `AgentLoopManager` / `AgentLoopWorker` 的整条控制流。
- 首版 bridge reference implementation：`ReactAgentLoop`
- 后续 validation：
  - `Retool`
  - `SWE-Agent`

### 2.8 Postprocess / assembler 策略

- 新架构引入独立的 `TrajectoryAssembler`。
- 其职责是：`trajectory -> DataProto`
- 行为上对齐现有 `AgentLoopWorker` 的训练可观察输出契约。
- 不直接复用旧 `AgentLoopWorker` 的整套 postprocess 栈。

原因：

- 旧 `AgentLoopWorker` postprocess 混合了：
  - 核心训练样本组装
  - reward / teacher 运行时接线
  - legacy `extra_fields` 兼容层
- 新架构应对齐输出契约，而非机械继承旧运行时控制流。

### 2.9 双重 trajectory bookkeeping

- 双重 bookkeeping 只适合作为迁移 / debug 期的 shadow verification 手段。
- 不进入主线运行时设计。
- 正式运行时的 token-level trajectory 真相源统一收敛到 `Gateway`。

### 2.10 fully_async_policy 兼容边界

- `fully_async_policy` 当前建立在现有 `AgentLoopManager/Worker` 体系之上，并通过 `FullyAsyncLLMServerManager` 支持 partial rollout resume。
- 首版 `AgentFramework/Gateway` 不要求额外适配 `fully_async_policy`。
- 如果首版不把 `InferenceRuntime` ownership 上提，则 `fully_async_policy` 应保持不变。
- 若未来抽取共享 runtime，抽取结果必须允许 fully-async 路径继续使用 specialized `LLMServerManager` 行为。
- 结论：
  - `fully_async_policy` 是 runtime 抽取的兼容约束。
  - 它不是 `AgentFramework` 首版阻塞项。

## 3. 模块落位

建议新增模块：

- `verl/experimental/agent_gateway/`
  - `AgentGatewayManager`
  - `GatewayActor`
  - `GatewaySession`
  - `Trajectory`
  - session/trajectory state types
- `verl/experimental/agent_framework/`
  - `AgentFramework`
  - `AgentFrameworkManager`
  - `FrameworkWorker`
  - `VerlLoopFramework`
  - `TrajectoryAssembler`

现有模块处理：

- `verl/experimental/agent_loop/agent_loop.py`
  - 保留 `AgentLoopManager` / `AgentLoopWorker` / `AgentLoopBase`
  - 当前工作流不主动重构其 runtime ownership
  - 后续由上游 rollout runtime 抽取统一收口
- `recipe/...`
  - 首版不重组目录
  - 通过 `VerlLoopFramework` 进行 bridge

上游/后续模块：

- `verl/experimental/inference_runtime/`
  - 属于共享 rollout runtime 的理想落位
  - 但不作为本工作流的首个自有 PR 交付项

## 4. 关键组件职责

### 4.1 共享 rollout runtime / builder seam

负责：

- rollout replicas 初始化
- `server_handles` / `server_addresses`
- `GlobalRequestLoadBalancer`
- `AsyncLLMServerManager`
- 相关 lifecycle 接口

不负责：

- framework worker 编排
- gateway session 路由
- DataProto 组装

ownership 目标：

- 长期应由 trainer/factory 层创建并注入到 old/new rollout path。

开发期允许：

- 由 `AgentFrameworkManager` 通过单一 builder/seam 暂时创建。
- 这个 seam 可以只是 manager 内的私有 helper/factory，不要求先抽成独立模块。
- 该过渡方案不得改变 future extraction 的可替换性。
- `AgentLoopManager` 与 `fully_async_policy` 在本工作流中不要求同步改造。

### 4.2 AgentFrameworkManager

负责：

- 创建并持有 `FrameworkWorker`
- 创建并持有 Gateway 子系统
- 依赖共享 rollout runtime（当前可经 manager-local seam 获取）
- batch chunk 分发与结果聚合
- 汇总 timing / metrics

不负责：

- 单 session 执行
- trajectory 组装
- rollout runtime 初始化细节

### 4.3 FrameworkWorker

负责：

- 持有共享对象：
  - tokenizer / processor
  - framework 实例
  - gateway manager client
  - 可选 reward / teacher hooks
- 在 worker 内并发处理多个 sample/session
- 调用 framework 单 session 流程
- 调用 `TrajectoryAssembler`

### 4.4 AgentFramework

负责：

- session lifecycle orchestration
- `run_session`
- `compute_reward`
- 将 trajectories 与 rewards 组织为 assembler 输入

不负责：

- cluster-level worker orchestration
- rollout runtime 初始化

### 4.5 VerlLoopFramework

负责：

- bridge 现有 `AgentLoopBase` 风格 agent
- 优先复用 agent-specific 行为
- 避免旧 manager/worker 控制流进入新架构主路径

首版原则：

- 可以保守复用部分 `AgentLoopBase.run()`
- 但不复用旧 `AgentLoopWorker` 的整条 postprocess / reward / teacher 栈
- Gateway 是 trajectory 真相源

### 4.6 AgentGatewayManager

建议作为 Ray actor。

最小接口：

- `create_session(session_id, metadata=None) -> GatewaySession`
- `finalize_session(session_id) -> list[Trajectory]`
- `abort_session(session_id) -> None`
- `wait_for_completion(session_id, timeout=None) -> None`

只保存 control-plane 状态：

- `gateway_handles`
- `session_id -> gateway`
- `active_sessions_per_gateway`
- 轻量 debug counters

不保存：

- 实际 session message / token / trajectory 内容
- tokenizer / processor
- rollout backend 细节

### 4.7 GatewayActor

一个 actor 持有多个 `SessionState`。

首版 `SessionState` 建议字段：

- `session_id`
- `metadata`
- `normalized_messages`
- `trajectories`
- `current_trajectory`
- `completion_event`
- `reward_info`
- `completed`
- `aborted`
- `created_at`
- `updated_at`
- `lock`

首版 `TrajectoryState` 建议字段：

- `trajectory_id`
- `prompt_ids`
- `response_ids`
- `response_logprobs`
- `loss_mask`
- `num_turns`
- `closed_reason`

首版不默认维护：

- `message_hashes`
- 完整 `turn_records`

## 5. Postprocess / Assembler 设计

### 5.1 必须严格对齐的训练契约

主字段：

- `prompts`
- `responses`
- `response_mask`
- `input_ids`
- `attention_mask`
- `position_ids`
- 可选 `rollout_log_probs`
- `rm_scores`
- `__num_turns__`

行为规则：

- prompt 左 padding
- response 右 padding
- `response_mask` 中 LLM 生成 token 为 `1`
- `rm_scores` 写在最后一个有效 response token 上

### 5.2 可以抽成共享 helper 的纯逻辑

- padding helper
- `response_mask` / `attention_mask` 组合规则
- `position_ids` 计算 helper
- `rm_scores` helper
- 纯 `TensorDict` / `DataProto` 构造 helper

### 5.3 不进入首版核心 assembler 的逻辑

运行时接线 / legacy compatibility：

- reward loop RPC 调度
- teacher logprob 计算与调度
- legacy `extra_fields` 全量稳定键补齐
- `reward_extra_keys`

### 5.4 首版可作为 optional enricher 的能力

- `rollout_log_probs`
- `routed_experts`
- 必要的 multimodal batch 补充

其中：

- `routed_experts` 如果 backend 稳定支持，应在首版 capability 范围内支持。
- 它不应被当作主路径必经字段，但可以在 assembler 中作为 optional enricher 写入 batch。

## 6. 测试与验收

### 6.1 组件测试

至少覆盖：

- `AgentGatewayManager` session routing
- `GatewayActor` create / finalize / abort / complete / wait
- prefix match / mismatch 行为
- 单 session 串行与跨 session 并发语义
- `TrajectoryAssembler` 主字段组装
- `routed_experts` optional capability（若 backend 支持）

### 6.2 Bridge 等效性测试

首版仅围绕 `ReactAgentLoop`。

比较旧路径与新路径的训练可观察输出：

- `prompts`
- `responses`
- `response_mask`
- `input_ids`
- `attention_mask`
- `position_ids`
- `rollout_log_probs`
- `rm_scores`
- `__num_turns__`
- `routed_experts`（若支持）

### 6.3 轻量 E2E / inspection run

本系列首个功能 PR 必须包含一次轻量但真实的端到端训练或 rollout 验证，并提供 reward 曲线。

要求：

- 可以使用小模型 / 小数据 / 小步数
- 但必须展示 reward 曲线趋势与基本可学性
- 以 `ReactAgentLoop` bridge 路径为主 acceptance scenario

后续 validation：

- `Retool`
- `SWE-Agent`

## 7. PR 分期

### 上游前置项：共享 rollout runtime 抽取（非本系列 PR）

交付：

- rollout runtime seam
- `GlobalRequestLoadBalancer`
- `AsyncLLMServerManager`
- runtime lifecycle 接口
- 现有 `AgentLoopManager` 依赖共享 runtime
- 保持 `fully_async_policy` specialized manager 行为不变

不交付：

- Gateway / Framework
- bridge
- 行为语义变化

说明：

- 该部分由上游推进，不作为当前工作流的独立 PR 目标。
- 当前工作应只预留清晰 seam，避免未来替换成本过高。

### 本系列 PR1：Gateway + Framework 主闭环

交付：

- `AgentGatewayManager` / `GatewayActor`
- `AgentFrameworkManager` / `FrameworkWorker` / `AgentFramework`
- `TrajectoryAssembler`
- `ReactAgentLoop` reference bridge
- 组件测试
- bridge 等效性测试
- 轻量 E2E + reward 曲线
- optional capability：`routed_experts`（若 backend 支持）
- manager-local runtime seam（作为过渡实现）

### 本系列 PR2：bridge 补齐与 validation

交付：

- 更完整 legacy/bridge glue logic
- `Retool` validation
- 需要时补更多 capability
- `SWE-Agent` 作为进一步 validation，而非首版阻塞项

## 8. AgentLoop 与 AgentFramework 功能对照

| 维度 | 现有 AgentLoop | 计划中的 AgentFramework | 首版状态 |
| --- | --- | --- | --- |
| 管理器关系 | `AgentLoopManager` 独立 | `AgentFrameworkManager` 与其并列 | 已收敛 |
| 共享 rollout runtime | 内嵌于 `AgentLoopManager` | 长期抽到共享 runtime | 上游前置项 |
| 运行时分层 | `Manager -> Worker -> AgentLoopBase` | `Manager -> FrameworkWorker -> AgentFramework` | 已收敛 |
| LLM sticky routing | `GlobalRequestLoadBalancer` + `AsyncLLMServerManager` | 首版通过 manager-local seam 复用现有路径 | 本系列 PR1 |
| Gateway session routing | 无 | `AgentGatewayManager` + `GatewayActor` | 本系列 PR1 |
| agent session lifecycle | 分散于 `AgentLoopBase.run()` | 收敛到 `AgentFramework` | 本系列 PR1 |
| trajectory 采集 | agent loop 内部各自维护 | Gateway 统一维护 | 本系列 PR1 |
| prefix mismatch | 依 agent 实现 | Gateway 统一切新 trajectory | 本系列 PR1 |
| DataProto 主字段 | `AgentLoopWorker` postprocess | `TrajectoryAssembler` 对齐输出契约 | 本系列 PR1 |
| `rollout_log_probs` | 已支持 | 首版支持 | 本系列 PR1 |
| `routed_experts` | optional 支持 | 作为 optional enricher 支持 | 本系列 PR1（条件项） |
| reward loop wiring | `AgentLoopWorker` 内部调度 | 通过 framework hook 接入 | 本系列 PR2/后续 |
| teacher logprobs | `AgentLoopWorker` 生态中支持 | 后续补齐 | 本系列 PR2/后续 |
| multimodal batch 补充 | 已有路径 | 基础兼容 + 后续补齐 | 本系列 PR1/PR2 |
| `extra_fields` 全量兼容 | 已有 | 首版仅必要项 | 本系列 PR2 |
| reference implementation | 多个现有 agent loop | `ReactAgentLoop` bridge | 本系列 PR1 |
| workload validation | 现有 tool/retool/SWE 等 | `Retool` / `SWE` 后续 validation | 本系列 PR2/后续 |
| fully_async_policy | `FullyAsyncLLMServerManager` + specialized AgentLoop path | 首版不适配，保持兼容 | 上游约束 / 后续议题 |

## 9. 待后续补齐的明确能力

- teacher logprobs / distillation 路径
- 全量 legacy `extra_fields` 兼容
- 更完整 multimodal 兼容
- 更多 `AgentLoopBase` 子类 bridge
- `Retool` validation
- `SWE-Agent` validation
- `AgentFramework` fully-async rollout 支持

## 10. 本文档对应的实施原则

- 先把边界设计清楚，再交付 Gateway/Framework 主能力。
- 最终目标是 trainer-owned shared runtime，但当前工作流允许 manager-local seam 作为过渡。
- 对齐训练可观察输出契约，而非机械复制旧控制流。
- Gateway 是 token-level trajectory 真相源。
- 双重 bookkeeping 仅用于迁移期验证，不进入主线。
- 首版需包含可审查的 E2E reward 曲线证据。
