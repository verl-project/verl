# verl 深度学习教程总览

这套中文教程不是为了重复官方文档，而是为了补上最容易“卡脑子”的那一层：

- 命令为什么能跑起来？
- 控制流和计算流到底是怎么拆开的？
- 显存、权重重分片、跨进程通信到底卡在哪？
- PPO、GRPO、KL 控制、Rollout Correction 在源码里分别落在哪几个函数上？

如果你已经会看 Python、会改 Hydra 配置、也能大致理解 RLHF，但一打开 `verl` 目录仍然觉得“东西很多却抓不住主轴”，这套文档就是给你的。

## 建议阅读顺序

1. [`quick-start.md`](./quick-start.md) -- 先把第一次跑通时的心智模型搭起来。
2. [`architecture.md`](./architecture.md) -- 看清 HybridFlow、资源池、角色分工与后端切换。
3. [`source-code-tour.md`](./source-code-tour.md) -- 沿着 `main_ppo.py -> RayPPOTrainer -> worker` 一路对线源码。
4. [`math-theory.md`](./math-theory.md) -- 把 PPO、GRPO、KL 和 rollout correction 彻底拆成人话。

```{toctree}
:maxdepth: 1

quick-start
architecture
source-code-tour
math-theory
```

```{mermaid}
flowchart TD
    A[快速上手<br/>先跑通第一条训练链路] --> B[架构深潜<br/>HybridFlow、拓扑与后端]
    B --> C[源码导览<br/>从入口一路读到 update_actor()]
    C --> D[数学拆解<br/>PPO、GRPO、KL、三策略修正]
    B --> E[官方文档与示例<br/>继续往细节深挖]
```

## 一句话先讲透 `verl`

`verl` 是一个给大语言模型做 RL 后训练的系统。它最核心的设计不是“又实现了一个 PPO”，而是 **HybridFlow**：

- **控制流** 放在一个相对轻量的单控制器里
- **计算流** 交给分布式 worker 去干

这样做的好处是：算法逻辑仍然能写得像普通 Python 程序一样清晰，但底层计算却可以自由切换到 FSDP、Megatron、vLLM、SGLang、TRT-LLM 等后端。

你可以把它想成一个片场：

- 单控制器像导演，负责安排“下一幕该拍什么”
- worker 像摄影组、灯光组、后期组，负责真正耗资源的活
- `DataProto` 就像在各个组之间来回流转的器材箱和素材箱

## 先记住这四个关键概念

### 1. `DataProto` 是全系统的“快递箱”

`verl/protocol.py` 里的 `DataProto` 负责在 rollout、reward、reference log-prob、critic value、advantage、actor update 等环节之间搬运数据。你只要把它看懂，整个系统的数据血管就通了。

### 2. `@register` 不是装饰品，而是分布式 RPC 合约

在 `verl.single_controller` 里，一个被 `@register` 标记的方法，不只是“可以被调用”。它还声明了：

- 输入怎么切
- 哪些 rank 真正执行
- 输出怎么收回来

所以控制器才可以像调普通对象一样去调 `worker_group.generate_sequences(batch)`，但底层其实已经做了分发、执行和汇聚。

### 3. role + resource pool 是资源拓扑语言

训练器思考的是逻辑角色：

- actor / rollout
- ref policy
- critic
- reward

Ray resource pool 决定这些角色物理上放在哪些 GPU 上。改映射，不改算法主循环，这就是 `verl` 伸缩性强的根源之一。

### 4. 主训练循环故意保持“薄”

`verl/trainer/ppo/ray_trainer.py` 很重要，但它不是把所有重算子都塞在里面。它主要做编排：

- 调 rollout
- 调 reward
- 调 KL
- 算 advantage
- 调 actor / critic update

真正底层的模型计算在 worker 里，核心数学在 `verl/trainer/ppo/core_algos.py` 里。

## 快速术语表

| 术语 | 在 `verl` 里的现实含义 |
| --- | --- |
| HybridFlow | 把 RL 控制流和分布式计算流拆开。 |
| Single controller | 一个单独的 Python 进程，负责训练编排。 |
| WorkerGroup | 把一组远端 worker 包成一个“像本地对象一样调用”的门面。 |
| ResourcePool | 一块被切出来分配给某类角色的集群资源。 |
| HybridEngine | actor 和 rollout 共置/协同的执行形态，用来减少切换成本。 |
| 3D-HybridEngine | 面向 Megatron 场景的重分片设计，降低训练与生成切换的显存和通信代价。 |
| Rollout correction | 把行为策略、近端锚点策略、当前策略拆开的三策略修正框架。 |

## 贯穿全套教程的源码锚点

- 入口：`verl/trainer/main_ppo.py`
- 主训练循环：`verl/trainer/ppo/ray_trainer.py`
- 核心数学：`verl/trainer/ppo/core_algos.py`
- 数据协议：`verl/protocol.py`
- 单控制器机制：`verl/single_controller/base/*`、`verl/single_controller/ray/base.py`
- FSDP worker：`verl/workers/fsdp_workers.py`
- Megatron worker：`verl/workers/megatron_workers.py`
- rollout 后端：`verl/workers/rollout/*`
- reward 系统：`verl/trainer/ppo/reward.py`、`verl/workers/reward_manager/*`、`docs/advance/reward_loop.rst`

## 建议同步打开的官方文档

- `docs/hybrid_flow.rst`
- `docs/single_controller.rst`
- `docs/examples/ppo_code_architecture.rst`
- `docs/workers/ray_trainer.rst`
- `docs/workers/fsdp_workers.rst`
- `docs/workers/megatron_workers.rst`
- `docs/algo/ppo.md`
- `docs/algo/grpo.md`
- `docs/algo/rollout_corr_math.md`

## 下一步

如果你想最快从“目录很多很乱”进入“我已经抓住主线了”，下一篇请先读 [`quick-start.md`](./quick-start.md)。
