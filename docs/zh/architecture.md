# 架构深潜：HybridFlow、资源拓扑与后端切换

这篇想回答一个很多人第一次读 `verl` 都会遇到的问题：

> 为什么这个仓库看起来同时像训练框架、推理框架、Ray 编排框架和算法库？

答案是：它本来就在同时解决这几层问题，但它不是把它们揉成一团，而是尽量把它们拆开。

## 1. `verl` 想解决的，不只是“写一个 PPO”

到了大模型 RL 后训练场景，系统真正难的地方通常有五个：

1. **控制流很复杂** -- rollout、reward、KL、value、update、validation、checkpoint 都要排顺序
2. **后端异构** -- 训练也许想用 FSDP 或 Megatron，生成却更适合 vLLM / SGLang / TRT-LLM
3. **显存容易重复占用** -- actor、rollout、ref、critic、optimizer state、KV cache 都在抢 HBM
4. **资源拓扑不固定** -- 有时想共置，有时想拆池，有时 reward/teacher 还要单独拿资源
5. **算法变化很快** -- PPO、GRPO、rollout correction、reward loop 等组合经常变化

如果把这些东西全部揉进一个巨大的分布式脚本里，短期可能快，长期通常会越来越难维护、难扩展、难 debug。

## 2. HybridFlow 的核心思想：脑力活和体力活分开

`verl` 的核心设计可以浓缩成一句话：

> 把 RL 的**控制流**留在一个可读的单控制器里，把真正吃资源的**计算流**下放给分布式 worker。

```{mermaid}
flowchart LR
    subgraph Controller[单控制器进程]
        A[读 batch]
        B[调 rollout]
        C[调 reward / ref / critic]
        D[算 advantage]
        E[调 actor / critic update]
    end

    subgraph Workers[GPU 上的分布式 worker group]
        F[Actor / Rollout]
        G[Reference Policy]
        H[Critic]
        I[Reward]
    end

    A --> B --> C --> D --> E
    B <--> F
    C <--> G
    C <--> H
    C <--> I
```

你可以把它想成片场分工：

- 控制器像导演，负责排场次和下指令
- worker 像摄影组、录音组、后期组，负责真正重的工作
- `DataProto` 就像在各个组之间流转的器材箱 / 素材箱

这也是为什么 `docs/hybrid_flow.rst` 与 `docs/single_controller.rst` 是读懂整个系统最重要的两篇官方文档。

## 3. 这个仓库可以按四层来读

### 第一层：入口与编排层

回答“下一步做什么”的文件：

- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/utils.py`

### 第二层：数据搬运与 RPC 抽象层

回答“一次控制器调用怎么变成很多远端 worker 一起干活”的文件：

- `verl/protocol.py`
- `verl/single_controller/base/decorator.py`
- `verl/single_controller/base/worker.py`
- `verl/single_controller/base/worker_group.py`
- `verl/single_controller/ray/base.py`

### 第三层：后端 worker 与引擎层

回答“真正的模型计算在谁手里”的文件：

- `verl/workers/fsdp_workers.py`
- `verl/workers/megatron_workers.py`
- `verl/workers/engine/base.py`
- `verl/workers/rollout/*`
- `verl/workers/sharding_manager/*`

### 第四层：RL 数学与 reward 层

回答“到底在优化什么目标”的文件：

- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/reward.py`
- `verl/workers/reward_manager/*`
- `docs/advance/reward_loop.rst`

## 4. 为什么 `DataProto` 是全场最重要的数据结构之一？

很多人第一次看源码，会把 `DataProto` 当成“一个带点花活的 batch dict”。这会低估它。

它实际上是整个训练链路的统一数据容器，至少承载三类信息：

- `batch`：张量型数据，比如 token、mask、log-prob、value、reward、advantage
- `non_tensor_batch`：字符串、uid、多模态补充信息等非张量数据
- `meta_info`：温度、global step、timing、token 统计等运行期元信息

如果没有这个统一容器，每走到一个阶段就要重新约定一套输入输出格式，系统会立刻变得很难维护。

## 5. `@register` 的真正作用：把普通函数变成分布式协议

在 `verl.single_controller` 里，一个 worker 方法被 `@register` 装饰后，声明的不是“它能被调用”，而是：

- 输入该怎么拆
- 哪些 rank 该执行
- 结果该怎么收回来

所以控制器这一句：

```python
output = actor_rollout_wg.generate_sequences(batch)
```

底下可能已经完成了完整的“切分 -> 分发 -> 执行 -> 汇聚”。

```{mermaid}
flowchart TD
    A[控制器调用 worker-group 方法] --> B[@register 元信息]
    B --> C[dispatch 函数<br/>按 DP 或 Megatron 拓扑切输入]
    C --> D[execute 函数<br/>选择哪些 rank 真执行]
    D --> E[collect 函数<br/>把结果重新并回一个 DataProto]
```

这是 `verl` 很漂亮的一点：算法主循环依然像单机程序一样清晰，但底层已经是拓扑感知的分布式执行。

## 6. role、resource pool、placement：逻辑角色与物理资源分离

训练器思考的是逻辑角色：

- actor / rollout
- reference policy
- critic
- reward
- 某些场景下还有 teacher 或 reward loop worker

而 resource pool 定义的是这些角色物理上落在哪些资源上。

```{mermaid}
flowchart LR
    subgraph Roles[逻辑角色]
        A[Actor / Rollout]
        B[Reference]
        C[Critic]
        D[Reward]
    end

    subgraph Pools[物理资源池]
        P1[global_pool<br/>若干节点与 GPU]
        P2[reward_pool<br/>可选独立资源]
    end

    A --> P1
    B --> P1
    C --> P1
    D --> P1
    D -. 可选拆池 .-> P2
```

这套抽象的厉害之处在于：

- 想共置，就让多个 role 共享一个 pool
- 想拆开，就改 role 到 pool 的映射
- 算法主循环本身几乎不用重写

## 7. FSDP 和 Megatron：外层循环相似，内层机械结构差很多

从控制器角度看，PPO 主循环几乎一样；从 worker 角度看，两条路差别很大。

| 维度 | FSDP 路径 | Megatron 路径 |
| --- | --- | --- |
| 主要 worker 文件 | `verl/workers/fsdp_workers.py` | `verl/workers/megatron_workers.py` |
| 更适合什么 | 原型验证、快速扩展、Hugging Face 模型接入 | 大规模训练、复杂并行、高吞吐 |
| 并行复杂度 | 心智负担相对小 | TP / PP / DP / CP / EP 全都可能上场 |
| 重分片成本 | 机制更直接，但大规模下可能更重 | 更复杂，但更有机会把规模做上去 |
| 研究友好度 | 更高 | 更低 |

最重要的不是“谁更好”，而是：

> **外层算法编排尽量不变，底层计算后端可以换。**

## 8. 为什么 3D-HybridEngine 这么关键？

训练和生成如果共享或部分共享硬件，一个很容易踩的坑就是状态重复占用：

- actor 权重一份
- rollout 权重再来一份
- KV cache 一份
- optimizer / activation 再来一份

显存就是这么被吃没的。

一个很直观的近似理解是：

- **朴素设计的峰值显存** 近似像 `actor 权重 + rollout 权重 + KV cache + 训练附加状态`
- **HybridEngine / 3D-HybridEngine 想达到的目标** 更接近 `共享或重分片后的权重 + 切换开销 + KV cache + 必要训练状态`

这不是严谨公式，但非常有助于抓住设计动机。

### 一个“算给人看”的玩具例子

假设一张卡预算是 80 GB：

- actor 权重 28 GB
- rollout 权重 28 GB
- KV cache 18 GB
- 训练额外状态 10 GB

如果粗暴复制，需求变成：

`28 + 28 + 18 + 10 = 84 GB`

已经超了。

如果系统能把 actor / rollout 的权重做共享或重分片，让切换只付出一个相对小的中间开销，比如：

- 共享后的主体权重 28 GB
- 切换额外开销 6 GB
- KV cache 18 GB
- 训练额外状态 10 GB

那峰值大致就会变成：

`28 + 6 + 18 + 10 = 62 GB`

这时候不但能跑，还留出了调优空间。

实际显存取决于后端、batch、并行配置、offload 等细节，但这个算例已经足够说明：为什么 `verl` 把训练到生成的切换成本当成头等公民。

## 9. 一步训练从系统角度到底长什么样？

```{mermaid}
flowchart TD
    A[加载 prompt batch] --> B[按 rollout.n 重复采样]
    B --> C[用 rollout 引擎生成 response]
    C --> D[可选：重算 old log-probs]
    D --> E[可选：算 reference log-probs]
    E --> F[可选：算 critic values]
    F --> G[抽取规则奖励或模型奖励]
    G --> H[套 KL penalty / KL loss 逻辑]
    H --> I[计算 advantage]
    I --> J[更新 critic]
    I --> K[更新 actor]
    J --> L[checkpoint / validation / logging]
    K --> L
```

这里面谁重、谁轻，其实非常清楚：

- 重：rollout、log-prob、value、actor update、critic update
- 轻：编排、reward 组合、advantage 计算、指标汇总

这也正是为什么控制器和 worker 必须拆开。

## 10. 出现瓶颈时该先读哪里？

| 症状 | 优先读什么 |
| --- | --- |
| rollout 很慢 | `verl/workers/rollout/*`、`docs/workers/sglang_worker.rst`、rollout 配置说明 |
| rollout 阶段显存爆 | `docs/workers/megatron_workers.rst`、`docs/workers/fsdp_workers.rst`、`verl/workers/sharding_manager/*` |
| batch / rank / 拓扑看不明白 | `docs/single_controller.rst`、`verl/single_controller/base/decorator.py` |
| reward 很难扩展 | `docs/advance/reward_loop.rst`、`verl/workers/reward_manager/*` |
| 数学看起来不对劲 | `verl/trainer/ppo/core_algos.py`、`docs/algo/ppo.md`、`docs/algo/grpo.md`、`docs/algo/rollout_corr_math.md` |

## 11. 最后只记一句话也够

这套架构最关键的一句话是：

> **`verl` 努力让算法主循环保持稳定，同时让 placement、并行方式和后端实现可以在底下自由切换。**

这就是它为什么既能服务“单卡算法原型”，也能服务“多节点大规模 RL 后训练”的根本原因。

下一篇建议看 [`source-code-tour.md`](./source-code-tour.md)，把这张系统骨架和真正的文件路径一一对应起来。
