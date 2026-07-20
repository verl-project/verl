# Fully-Async 训练的动态资源伸缩

本模块为 fully-async 训练框架提供**混合推理资源的动态伸缩**能力：让 Trainer 节点的 GPU 在训练空闲期参与 rollout 生成，从而提升整体 GPU 利用率。

> **速览**
> - 两种推理副本：**Standalone**（专用 rollout 节点，常驻在线）+ **Hybrid**（与 Trainer 节点 GPU 共享，训练空闲时激活，训练前 offload 权重并归还显存）。
> - 由可插拔的 **Policy** 决定何时激活/停用 Hybrid 副本；`DynamicResourceController` 管理生命周期（状态机 `STANDALONE_ONLY <-> HYBRID_ACTIVE`）。
> - 当 Standalone 产能充足时，Hybrid 直接跳过 rollout 专心训练，避免无谓的模式切换；权重同步也可按需跳过 Hybrid，节省通信。
> - 实测（Qwen3.5-35B-A3B / DAPO-Math-17K）：端到端训练时间快约 **15.3%**，reward 曲线与 baseline 一致。

---

## 1. 概览

### 问题

在 fully-async 的分离架构下，Trainer 节点的 GPU 在等待 rollout 数据时空闲，Standalone Rollout 节点在训练阶段也空闲。两侧 GPU 利用率都不高。

### 方案

**Hybrid + Standalone 双模推理资源**设计：

- **Standalone replicas**：部署在专用 rollout 节点上的常驻推理副本。
- **Hybrid replicas**：与 Trainer 节点 GPU 共享的推理副本。在训练空闲期被激活参与 rollout；每个训练 step 开始前，offload 权重并把显存归还给训练引擎。

![动态资源架构](https://github.com/zpltys/Blob/blob/main/dynamic_resource.png?raw=true)

上图展示了三个连续训练 step 的**资源-时间布局**。纵轴是 GPU 资源，分为两个池：

- **Standalone Resource**（上，GPU 0–n）：专用 rollout GPU，在所有 step 的 *Rollout* 阶段（蓝色）持续工作，在 *Trainer* / *Weight Sync* 阶段空闲。
- **Hybrid Resource**（下，GPU n+1–n+m）：Trainer 节点 GPU，在 rollout 与训练之间切换。*Rollout* 模式下加入 Rollout LoadBalancer 一起生成样本（蓝）；*Trainer* 模式下执行 PPO mini-batch 更新（黄）；在每个 *Weight Sync* 边界把最新权重广播给所有副本。

横轴三个 step 体现了 Hybrid 的不同行为：

- **Step 1 & 2**：step 开始时 MessageQueue 还没攒够样本，Hybrid 资源进入 *Rollout* 模式，与 Standalone 一起生成样本；攒够后切回 *Trainer*。
- **Step 3**：step 开始前 MessageQueue **已有足够样本**（来自前面 step 的缓冲），Hybrid 直接进入 *Trainer* 模式、**跳过 rollout**——这是关键优化：当 Standalone 产能充足时，Trainer GPU 专心训练，避免无谓的模式切换开销。

切换阈值由 `dynamic_schedule_deactivate_ratio` 控制：当收集到的样本数达到 `deactivate_ratio × required_samples × trigger_parameter_sync_step` 时，控制器停用 Hybrid 副本。中心化的 *Rollout LoadBalancer* 把 MessageQueue 的生成请求分发到所有 active 副本。

两个请求处理机制保证模式切换平滑：

- **Hybrid → Trainer（停用）**：Hybrid 从 Rollout 切回 Trainer 时，其上**在途请求（in-flight）被 abort**，并由 LoadBalancer **自动重分发到 Standalone** 继续完成 rollout。依赖 `FullyAsyncLLMServerClient` 的重试机制，对上层透明。
- **Trainer → Hybrid（激活）**：训练 step 结束后若 Hybrid 重新进入 Rollout 模式，`dynamic_schedule_enable_rebalance` 控制是否做一次 **reshuffle**：开启时清空 LoadBalancer 的 sticky-session 缓存、abort 所有 active 副本上的在途请求再 resume，请求按 least-loaded 重新路由，自然地把负载压向刚激活、在途为 0 的 Hybrid 副本。

- **Weight Sync（权重同步）**：Weight Sync 阶段先把权重从 Trainer 广播给所有 **Standalone** 副本（始终需要）；随后评估 policy 的 `should_activate_after_step()` 决定下一步是否激活 Hybrid。需要激活时**额外**把权重同步给 Hybrid 副本；否则**跳过**这第二次同步以节省通信开销——上图 Step 3 正是这种情况。

`DynamicResourceController` 管理 Hybrid 副本生命周期，可插拔的 **Policy** 决定何时激活/停用：

```
状态机：  STANDALONE_ONLY  <->  HYBRID_ACTIVE

Activate（weight sync 之后）：
  1. add_replicas               — 在 load balancer 中注册 hybrid 副本
  2. resume_generation_replicas — 允许 hybrid 副本接收请求

Deactivate（顺序至关重要）：
  1. remove_replicas  — 先切断路由，防止 retry loop 把请求重路由到正在下线的副本
  2. abort_replicas   — abort 在途请求，partial-rollout 的重试落到 standalone
  3. sleep_replicas   — 释放 KV cache + offload 权重，把 GPU 还给训练引擎
```

### 每个 training step 的 Policy 调用顺序

```
1. should_deactivate()          — 训练前；决定是否停用 hybrid 副本
2. deactivate_wait_samples()    — 若 (1) 为 True；返回最小缓冲样本阈值
3. should_activate_after_step() — weight sync 后；决定是否（重新）激活 hybrid 副本
4. request_rebalance()          — 激活后；跨副本重分发请求（若开启）
5. update_after_step()          — weight sync 后；更新 policy 内部状态
```

---

## 2. 配置参数

所有动态伸缩参数位于训练 config 的 `async_training` 段（`fully_async_ppo_trainer.yaml` 或 `fully_async_ppo_megatron_trainer.yaml`）。

### 核心参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `use_dynamic_resource_scheduling` | bool | `False` | 总开关。为 `True` 时，启动即在 Trainer 节点 GPU 上初始化 hybrid rollout 副本（sleeping，显存已归还训练引擎）。 |
| `dynamic_schedule_policy` | str | `"default"` | 策略名（`"default"` / `"static_fully_async"` / `"fixed_ratio"` / 自定义注册名）。 |
| `dynamic_schedule_deactivate_ratio` | float | `0.3` | 样本收集比例阈值。控制器等到 `deactivate_ratio × required_samples × trigger_parameter_sync_step` 个样本缓冲好再停用。越小越早停用；`1.0` 表示等满一个 batch。 |
| `dynamic_schedule_enable_rebalance` | bool | `True` | hybrid 激活后是否对在途请求做 rebalance（abort + 清 sticky cache + resume），按 least-loaded 路由。 |

### 复用的已有 async-training 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `staleness_threshold` | float | `0.1` | 允许的样本 staleness 比例；影响 `buffer_samples = expected × staleness_threshold`。 |
| `trigger_parameter_sync_step` | int | `4` | 两次 weight-sync 之间的 collection 数；用于上面的 deactivate wait 公式。 |
| `require_batches` | int | `1` | 每次 collection 的 ppo_mini_batch 数；决定 `required_samples`。 |


### Rollout 资源配置（`actor_rollout_ref.rollout` 下）

| 参数 | 说明 |
|------|------|
| `gpu_memory_utilization` | **hybrid** 副本（与训练共享 GPU）的显存占用，建议保持较低（如 0.3–0.5）。 |
| `standalone_gpu_memory_utilization` | **standalone** 副本（专用 rollout 节点）的显存占用（如 0.6–0.8）；为 `null` 时回退到 `gpu_memory_utilization`。 |

### Standalone 节点配置（`rollout` 下）

| 参数 | 说明 |
|------|------|
| `rollout.nnodes` | standalone rollout 节点数。设为 `0` 表示纯 colocated 模式（只有 hybrid，无专用 rollout 节点）。 |

---

## 3. 内置策略

### 3.1 `default` — DefaultDynamicSchedulePolicy

**文件：** `default_policy.py`

推荐策略，带**自适应 deactivate_ratio** 与**成本/收益门控**。

| 方法 | 行为 |
|------|------|
| `should_deactivate()` | 返回 `is_hybrid_active`（激活时即停用） |
| `deactivate_wait_samples()` | 返回 `deactivate_ratio × required_samples × trigger_parameter_sync_step` |
| `should_activate_after_step()` | 当生成样本落后于预期（存在缺口）**且**预估节省时间超过切换（activate+deactivate）成本时才激活 |
| `update_after_step()` | 自适应 `deactivate_ratio`（见下；`only_hybrid=True` 时禁用） |
| `request_rebalance()` | 清 sticky cache + abort 在途请求 + resume，按 least-loaded 重路由 |

**`update_after_step()` 的自适应逻辑**（以 `step_wait_samples` 为信号——真正等待新生成样本的数量，排除从队列积压中即时取走的样本）：

- 本 cycle 有真实等待样本（`total_wait_samples > 0`，说明 rollout 是瓶颈）：`ratio += clip(total_wait_samples / step_required_samples, 0.02, 0.1)`——缺口越大增量越大，单步夹在 [0.02, 0.1]，倾向于**更晚停用**。
- 本 cycle 无真实等待样本（说明训练是瓶颈）：`ratio -= 0.02`，倾向于**更早停用**。

**成本/收益门控**（`should_activate_after_step`）：重新激活要付出一次 activate + 一次 deactivate 的 `switch_cost`（最近 3 次 activate+deactivate 时长的滑动平均，无历史数据时用 10s 兜底）。仅当 standalone-only 产能缺口预估带来的收益 > `switch_cost` 时才激活。缺口→时间的换算用「单样本生成耗时」估计（`total_wait / total_wait_samples`），未观测到真实生成速率信号前用悲观的 `1000s` 初值兜底，倾向先激活直到真实数据收敛估计。

**适用场景：** Hybrid + Standalone 混部，追求最大 GPU 利用率。

### 3.2 `static_fully_async` — StaticFullyAsyncPolicy

**文件：** `static_fully_async_policy.py`

等价于原始 fully-async 策略，用于 baseline 对比或 colocated 回退：

| 方法 | 行为 |
|------|------|
| `should_deactivate()` | 返回 `is_hybrid_active` |
| `deactivate_wait_samples()` | **恒返回 `0`**（立即停用，不等待） |
| `should_activate_after_step()` | **恒返回 `False`**（weight sync 后不再激活） |
| `update_after_step()` | no-op |
| `request_rebalance()` | no-op（继承基类默认） |

**关键性质：**

1. **等价于标准 Fully-Async**：每个 step 开始立即停用 hybrid、永不重新激活，Trainer GPU 始终 100% 用于训练——与不开 `use_dynamic_resource_scheduling` 行为一致。
2. **Colocated 回退**：`rollout.nnodes=0` 时 `only_hybrid=True`，退化为经典 colocated 模式（训练 + 推理共享同一批 GPU，无独立 rollout 节点）。

### 3.3 `fixed_ratio` — FixedRatioDynamicSchedulePolicy

**文件：** `fixed_ratio_policy.py`

与 `default` 一致，但 `update_after_step()` 为 no-op——`deactivate_ratio` 全程保持初始值，不做自适应。`should_activate_after_step()` 也简化为纯缺口判断（不做成本/收益门控）。适合固定比例的对比实验。

> **注册提示：** `fixed_ratio` 目前未在 `__init__.py` 中导入，`@register_policy` 不会在默认 import 时触发。使用前需在 `__init__.py` 加一行 `from .fixed_ratio_policy import FixedRatioDynamicSchedulePolicy`，或在入口脚本手动 import。

---

## 4. 监控指标

动态资源伸缩在通用 `fully_async/*` 指标（见 [`docs/advance/fully_async.md`](../../../../docs/advance/fully_async.md)）之外，新增 `dynamic_resource/*` 系列指标，用于量化 Hybrid 与 Standalone GPU 的利用效果。本节给出每个指标的严格定义。

### 记号

| 符号 | 含义 |
|--------|---------|
| $a$ | `hybrid_gpus` = `trainer.nnodes × trainer.n_gpus_per_node`（Trainer 节点可在 rollout/train 间切换的 GPU 数） |
| $b$ | `standalone_gpus` = `rollout.nnodes × rollout.n_gpus_per_node`（专用 rollout 节点 GPU 数） |
| $\text{compute}_i$ | micro fit-step $i$ 的训练 GPU 计算时间（`timing_raw` 中 `reward`、`old_log_prob`、`ref`、`values`、`adv`、`update_critic`、`update_actor` 之和） |
| $\text{alloc}_i$ | micro fit-step $i$ 的分配 wall-clock 时间（从 `_fit_generate()` 到 `_fit_dump_data()`，含 `param_sync` 与激活开销） |
| $\text{cap}_k$ | rollout 区间 $k$ 的并发容量 = $\min(\text{servers}_k \times s,\ \text{maxReq})$ |
| $\text{active}_k$ | rollout 区间 $k$ 的活跃任务数 |
| sync cycle | 两次连续 parameter-sync 之间的窗口（即两次 `reset_staleness()` 之间 / 每 `trigger_parameter_sync_step` 次 collection），可能跨越多个 Trainer micro fit-step（含 partial-rollout 恢复） |

### 4.1 `dynamic_resource/train_resource_utilization`

整个 sync cycle 内，Trainer「训练回合」wall-clock 时间中真正用于训练 GPU 计算的占比。

对 cycle 内每个 micro fit-step $i = 1, \dots, n$：

- **分子**（$\text{compute}_i$）：`timing_raw` 训练计算 key 之和
  （`reward`、`old_log_prob`、`ref`、`values`、`adv`、`update_critic`、`update_actor`）。
  ——`ref` 是通过 `marked_timer(str(Role.RefPolicy), …)` 记录的 timing key（枚举的字符串值为 `"ref"`，而非 `"RefPolicy"`）。缺失的 key——如 `use_critic=False` 时的 `values`/`update_critic`——按 0 计。
- **分母**（$\text{alloc}_i$）：从 `_fit_generate()` 到 `_fit_update_weights()`、`_fit_dump_data()` 的 wall-clock（含 `timing_s/param_sync` 与 hybrid 激活开销，这些**故意不减**）

两者先对 cycle 内所有 micro-step 求和，再由总和求比（而非按 micro-step 平均）：

$$
U_{\text{train}} = \frac{\sum_{i=1}^{n} \text{compute}_i}{\sum_{i=1}^{n} \text{alloc}_i}
$$

未取比的原始分子/分母也直接作为 `dynamic_resource/train_compute_time_s` 和 `dynamic_resource/train_allocated_time_s` 上报（按 cycle 内 micro-step **求和**聚合），比值由 `MetricsAggregator` 从总和一次性计算，而非按 micro-step 平均。

### 4.2 `dynamic_resource/rollout_resource_utilization`

Rollouter 一步内（上一次到本次 `reset_staleness()` 之间的窗口）rollout 并发容量的实际使用占比。

Rollouter 记录 `(len(active_tasks), max_concurrent_samples)` 的事件驱动历史——每次提交样本、完成、暂停时 drain 时追加，**且**每当 `max_concurrent_samples` 本身变化（即动态伸缩下副本增减）时也追加——得到时间戳 $t_0 < t_1 < \dots < t_m$，每个区间 $[t_k, t_{k+1})$ 内有两个量保持恒定：活跃任务数 $\text{active}_k$ 和并发容量 $\text{cap}_k$（$t_0$ 为上一步结束/本步开始；$t_m$ 为「现在」，在 `reset_staleness()` 时追加，闭合窗口并包含尾部 drain/空闲时间）。

逐区间记录容量是因为动态伸缩下 $\text{cap}_k$ 在一个 step 窗口内**并非恒定**——副本可能中途激活/停用。若用窗口末尾单一容量近似整窗，会带偏容量不同的早期区间利用率，因此每个区间用各自的 $\text{cap}_k$。

每个区间的容量为 $\text{cap}_k = \min(\text{servers}_k \times s,\ \text{maxReq})$，其中 $\text{servers}_k$ 为区间 $k$ 的 `get_active_server_count()`，$s$ 为 `concurrent_samples_per_replica`，$\text{maxReq}$ 为 `max_required_samples`：

$$
\text{cap}_k = \min(\text{servers}_k \times s,\ \text{maxReq})
$$

每个区间的有效负载为 $\min(\text{cap}_k, \text{active}_k)$（容量 $\text{cap}_k$ 已经 clamp 过，实际服务并发数取容量与活跃任务数的较小值）。按**容量时间**加权（容量大的区间权重更高）：

$$
U_{\text{rollout}} = \frac{\sum_k (t_{k+1} - t_k) \cdot \min(\text{cap}_k, \text{active}_k)}{\sum_k (t_{k+1} - t_k) \cdot \text{cap}_k}
$$

无可积分时间跨度时（如总容量时间权重为 0；$\text{cap}_k = 0$ 的区间权重为 0 且被跳过）返回 `0.0`。

### 4.3 `dynamic_resource/resource_utilization`

sync cycle 的集群级 GPU 利用率，结合 Hybrid GPU 在 rollout/train 间的时间切分与 Standalone GPU（100% 时间用于 rollout）。

设 $x \in [0, 1]$ 为本 cycle wall-clock 中 Hybrid GPU 用于 rollout 的时间占比（其余 $1-x$ 用于训练），估计为「等待足够样本」时间之和（Hybrid-rollout 的 wall-clock，仅在本 cycle 停用 Hybrid 副本时记录）占 step 时间之和的比例：

$$
x = \mathrm{clip}\left(\frac{\text{wait}}{\text{step}},\ 0,\ 1\right)
$$

其中 $\text{wait}$ 为 `timing_s/wait_for_enough_samples` 之和，$\text{step}$ 为 `timing_s/step` 之和。本 cycle 从未把 Hybrid GPU 切入 rollout（如功能关闭、或跳过 `wait_for_enough_samples`）时 $x = 0$（Hybrid 时间 100% 用于训练）。

按各自测量的 GPU-秒 加权——Hybrid GPU 在 $(1-x) \cdot a$ GPU 时间内以 $U_{\text{train}}$ 训练、在 $x \cdot a$ GPU 时间内以 $U_{\text{rollout}}$ 做 rollout；Standalone GPU（$b$ 个）全部时间以 $U_{\text{rollout}}$ 做 rollout：

$$
U_{\text{cluster}} = \frac{(1-x) \cdot a \cdot U_{\text{train}} + (x \cdot a + b) \cdot U_{\text{rollout}}}{a + b}
$$

### 4.4 `dynamic_resource/mq_size`

Trainer `fit_step()`（含 partial-rollout 恢复）结束后，MessageQueue 中剩余样本数的**快照**（非平均）。若记该快照为 $Q$，则 $Q$ 等于 `message_queue_client.get_queue_size_sync()` 的返回值。按 cycle 内最后一次观测值上报，而非跨 micro-step 平均。

### 4.5 相关 Rollouter 侧指标：`fully_async/rollouter/step_generated_samples`

当前 param version 内 Rollouter 完整生成的样本数（每次 `reset_staleness()` 时重置为 0）。它与 `dynamic_resource/rollout_resource_utilization` 一起在 rollouter 的 `timing_raw` 中返回，是 Rollouter 追踪每步吞吐量的底层信号；因其与上述 `dynamic_resource/*` 指标在同一代码路径中计算，故在此一并说明。

### 注意事项

- **`rollout_resource_utilization` 的逐区间容量**：动态伸缩下容量 $\text{cap}_k$ 可能因副本激活/停用而中途变化；每个区间使用各自的 $\text{cap}_k$ 而非窗口末尾单一值，容量变化可被忠实反映。`max_concurrent_samples` 首次已知前记录的初始种子点（占位容量 0）会在真实容量可用后被修正。
- **`resource_utilization` 假设 `should_deactivate() == is_hybrid_active`**：$x$ 的估计依赖 `timing_s/wait_for_enough_samples` 忠实反映「本 cycle Hybrid GPU 处于 rollout 模式」，这对内置 `default`/`static_fully_async` 策略成立，但对激活语义不同的自定义策略未必成立。
- **`train_resource_utilization` 与 `rollout_resource_utilization` 不完全可比**：前者分母含 `param_sync`/激活开销（未减），后者分母是纯 rollout 窗口时间——解读合并后的 `resource_utilization` 时需注意。

---

## 5. 实验结果

### Benchmark：Qwen3.5-35B-A3B on DAPO-Math-17K

| 项 | 配置 |
|------|--------|
| 模型 | Qwen3.5-35B-A3B |
| 数据集 | DAPO-Math-17k (train) / AIME-2024 (val) |
| 后端 | Megatron（TP=4, PP=2, EP=8） |
| 硬件 | H20（8 GPUs/node） |
| 脚本 | `verl/experimental/fully_async_policy/shell/run_qwen35_35b_a3b_math_dynamic_megatron.sh` |
| 关键超参 | `dynamic_schedule_policy="default"`, `dynamic_schedule_deactivate_ratio=0.6`, `dynamic_schedule_enable_rebalance=True`, `staleness_threshold=0.5`, `trigger_parameter_sync_step=4`；hybrid `gpu_memory_utilization=0.45`，standalone `standalone_gpu_memory_utilization=0.7` |

**Baseline 1**：16 GPU 训练 + 16 GPU rollout（2 个训练节点 + 2 个 rollout 节点）。
**Baseline 2**：8 GPU 训练 + 24 GPU rollout（1 个训练节点 + 3 个 rollout 节点）。
**动态伸缩**：16 GPU 训练（2 节点）+ 16 GPU rollout（2 节点），Trainer GPU 上跑 Hybrid 副本。

#### 结果：端到端训练时间快约 15.3%

动态资源伸缩相比 8+24 baseline，总 wall-clock 训练时间降低约 **15.3%**，相比 16+16 baseline 降低约 **17.5%**，且 reward 曲线**完全一致**——证明分时共享 GPU 不损害训练质量。

**Reward 曲线**（动态伸缩 vs baseline）：

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_exp_reward.png?raw=true" width="600" />
</div>

两条曲线全程高度重合，说明动态伸缩不引入模型质量回归。

**每步运行时对比**：

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_exp_time.png?raw=true" width="600" />
</div>

每步运行时图显示，动态伸缩利用 Trainer GPU 在 rollout 空闲窗参与生成以降低 step 延迟，且随训练推进、policy 自适应 `deactivate_ratio`，差距逐渐拉大。

#### 资源利用率分解

三种 static 配置的差异在于**训练/推理负载配比**，下面的指标可以清楚地体现这一点。关键直觉：`dynamic_resource/mq_size`（每次 `fit_step()` 后 MessageQueue 中缓冲的样本数）直接反映瓶颈在生产端（rollout）还是消费端（trainer）。

- **16-16-static**（16 GPU 训练 + 16 GPU rollout）：trainer 消费样本的速度快于 16 GPU rollout 的生产速度，**瓶颈在生产端**——队列被持续抽到接近 0，trainer 一直等样本。
- **24-8-static**（8 GPU 训练 + 24 GPU rollout）：24 GPU rollout 的生产速度快于 8 GPU trainer 的消费速度，**瓶颈在消费（训练）端**——样本堆积，队列顶到容量上限。
- **dynamic**：根据实时队列深度在 rollout 和训练之间动态调配 Trainer 节点 GPU，使生产和消费匹配——队列保持适中水平，既避免了 16-16 的 trainer 空等，也避免了 24-8 的样本积压。

**MessageQueue 大小**（`dynamic_resource/mq_size`）：

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_exp_mq_size.png?raw=true" width="600" />
</div>

16-16-static 曲线贴近 0（生产瓶颈，trainer 空等）；24-8-static 曲线顶在容量上限（消费瓶颈，样本积压）；dynamic 曲线居中，维持生成/消费平衡。

**集群级 GPU 利用率**（`dynamic_resource/resource_utilization`）：

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_resource_utilization.png?raw=true" width="600" />
</div>

16-16-static 浪费训练侧 GPU 时间等样本；24-8-static 浪费 rollout 侧 GPU 时间过度生产未被消费的样本。动态伸缩让两侧 GPU 时间都落在有效计算上，集群级利用率最高。

**Rollout 侧利用率**（`dynamic_resource/rollout_resource_utilization`）：

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_rollout_resource_utilization.png?raw=true" width="600" />
</div>

16-16-static（生产瓶颈）的 16 个 rollout GPU 长期压满产能；24-8-static（产能过剩）的 24 GPU rollout 容量大量闲置，因为 trainer 消费不动。动态伸缩将 rollout 利用率维持在较高且稳定的水平——既不像 16-16 那样被压满到瓶颈，也不像 24-8 那样闲置。

**训练侧利用率**（`dynamic_resource/train_resource_utilization`）：

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_train_resource_utilization.png?raw=true" width="600" />
</div>

24-8-static 样本充足，trainer 从不等样本，训练侧利用率最高。16-16-static 的 trainer 大量时间花在等样本上（等待时间在该指标分母内，拉低了比值）。动态伸缩在样本充足时接近 24-8 的高水平，同时避免了 16-16 的空等浪费。

**负载配比权衡汇总**：

| 配置 | 瓶颈 | MQ 积压 | Rollout 利用率 | Train 利用率 |
|------|------|---------|---------------|-------------|
| 16-16-static | 生产端（rollout） | 贴近 0（trainer 空等） | 压满产能 | 低（等样本） |
| 24-8-static | 消费端（trainer） | 顶到上限 | 容量闲置 | 高（不等样本） |
| dynamic | 动态匹配 | 适中 | 高且稳定 | 接近 24-8 |

动态伸缩的价值：通过在训练和 rollout 之间动态调配 GPU 以匹配实际负载，同时避免 16-16 的 trainer 空等和 24-8 的样本积压，在不增加总 GPU 数的前提下获得更高的端到端吞吐。

---

## 6. 添加自定义策略

支持一个新策略分四步：

### Step 1：继承基类

```python
from verl.experimental.fully_async_policy.dynamic_schedule import (
    DynamicSchedulePolicyBase,
    DynamicScheduleContext,
    register_policy,
)

@register_policy("my_policy")
class MyDynamicSchedulePolicy(DynamicSchedulePolicyBase):

    def __init__(self, deactivate_ratio: float = 0.5, only_hybrid: bool = False):
        self.deactivate_ratio = deactivate_ratio
        self.only_hybrid = only_hybrid

    def should_deactivate(
        self,
        global_steps: int,
        is_hybrid_active: bool,
        ctx: DynamicScheduleContext,
    ) -> bool:
        """返回 True 则本步停用 hybrid 副本。"""
        return is_hybrid_active

    def deactivate_wait_samples(self, ctx: DynamicScheduleContext) -> int:
        """返回停用前需缓冲的最小样本数。"""
        return int(ctx.required_samples * ctx.trigger_parameter_sync_step * self.deactivate_ratio)

    def should_activate_after_step(
        self,
        global_steps: int,
        is_hybrid_active: bool,
        ctx: DynamicScheduleContext,
    ) -> bool:
        """返回 True 则 weight sync 后重新激活。"""
        return ctx.total_generated_samples < ctx.expected_samples + ctx.buffer_samples

    # 可选：覆盖以在每步后更新内部状态
    def update_after_step(self, global_steps: int, ctx: DynamicScheduleContext) -> None:
        pass

    # 可选：覆盖以自定义激活后的请求重分发
    def request_rebalance(self, global_steps: int, ctx: DynamicScheduleContext) -> None:
        pass
```

### Step 2：注册导入

把文件放进 `dynamic_schedule/`，并在 `__init__.py` 加一行导入：

```python
# dynamic_schedule/__init__.py
from .my_policy import MyDynamicSchedulePolicy
```

或在入口脚本手动 import 以触发 `@register_policy`。

### Step 3：在 config 中引用

```yaml
async_training:
  use_dynamic_resource_scheduling: True
  dynamic_schedule_policy: "my_policy"
  dynamic_schedule_deactivate_ratio: 0.5
  dynamic_schedule_enable_rebalance: True
```

### Step 4：使用 `DynamicScheduleContext` 字段

| 字段 | 类型 | 说明 |
|-------|------|------|
| `required_samples` | `int` | 每次 collection 的最小样本数（`ppo_mini_batch_size × require_batches`） |
| `trigger_parameter_sync_step` | `int` | 两次 weight-sync 之间的 collection 数 |
| `step_required_samples` | `int` | 派生字段，`required_samples × trigger_parameter_sync_step`，一个 weight-sync cycle 期望的总样本数 |
| `total_generated_samples` | `int` | 训练开始以来累计的 rollout 样本数 |
| `expected_samples` | `int` | 到当前 sync step 为止理论上需要的样本数 |
| `buffer_samples` | `int` | 允许的缓冲余量（`expected × staleness_threshold`） |
| `step_wait_times` | `list[float]` | 最新一步内每次 collection 的等待时间（秒） |
| `step_wait_samples` | `list[int]` | 最新一步内每次 collection 真正需要等待的样本数（`max(0, required_samples - collection 开始时队列大小)`），与 `step_wait_times` 并行；用于计算不被队列积压带偏的生成速率信号 |
| `only_hybrid` | `bool` | 无 standalone 副本时为 `True` |
| `last_activate_duration_s` | `float` | 上一次 activate 周期（weight sync + onload）时长，秒 |
| `last_deactivate_duration_s` | `float` | 上一次 deactivate 周期（offload）时长，秒 |

---

## 7. 文件结构

```
dynamic_schedule/
├── __init__.py                        # 公开导出 + policy 注册表
├── base.py                            # DynamicSchedulePolicyBase ABC, DynamicScheduleContext, registry
├── default_policy.py                  # DefaultDynamicSchedulePolicy（自适应动态伸缩）
├── static_fully_async_policy.py       # StaticFullyAsyncPolicy（原始 fully-async / colocated 回退）
├── fixed_ratio_policy.py              # FixedRatioDynamicSchedulePolicy（固定比例，无自适应）
├── dynamic_resource_controller.py     # DynamicResourceController（状态机 + 生命周期）
├── README.md                          # 指向 docs/advance/dynamic_resource.md 的指针
└── README_zh.md                       # 中文文档（本文件）
```
