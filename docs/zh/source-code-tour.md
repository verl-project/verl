# 源码导览：从 `main_ppo.py` 一路读到 `update_actor()`

这篇的目标很明确：给你一条**最省脑力的源码阅读路线**。

如果只记一条原则，请记这条：

> 读 `verl` 时，先从**控制器**往外读，再去看底层 worker。

否则你很容易一头扎进各种分布式细节，却始终抓不住“主流程到底是谁在指挥”。

## 1. 第一站：`verl/trainer/main_ppo.py`

这个文件是 PPO 类训练任务的总入口。

它在高层做的事情很简单：

1. 读取 Hydra 配置
2. 自动决定设备相关设置
3. 需要时初始化 Ray
4. 创建一个远端 `TaskRunner`
5. 让 `TaskRunner` 真正去搭训练任务

可以把它理解成：

```{mermaid}
sequenceDiagram
    participant U as 用户命令
    participant M as main_ppo.py
    participant R as Ray 运行时
    participant T as TaskRunner
    participant P as RayPPOTrainer

    U->>M: python -m verl.trainer.main_ppo
    M->>R: ray.init(...)
    M->>T: 创建远端 TaskRunner
    T->>P: 组装 trainer 与 worker 映射
    P->>P: 初始化 worker 并执行 fit()
```

这个文件之所以值得最先读，是因为它明确告诉你：

- 配置从哪进来
- 哪部分是本地单进程
- 哪部分开始变成远端分布式执行

## 2. `TaskRunner`：把配置翻译成拓扑

在 `main_ppo.py` 里，`TaskRunner` 是第一个真正重要的控制面对象。

它的工作不是直接训练模型，而是决定：

- 用哪类 worker
- 当前任务有哪些 role
- resource pool 怎么建
- critic / reward model / teacher model 到底需不需要

这一步就是从“配置文件世界”切进“运行时拓扑世界”。

常见分叉包括：

- FSDP / FSDP2 worker 路径
- Megatron worker 路径
- 当 `use_legacy_worker_impl` 关闭时的新 engine worker 路径

## 3. `RayPPOTrainer`：控制器的大脑

训练主控制器在：

- `verl/trainer/ppo/ray_trainer.py`

建议分三遍读。

### 第一遍：读构造函数和 dataloader

这一遍主要回答：

- 当前哪些 role 被启用
- 是否需要 reference policy、reward model、teacher、critic
- dataloader 怎么构建
- checkpoint / logger / profiler 这些钩子怎么准备

### 第二遍：读 worker 初始化

这一步最重要的思想是：

每个逻辑角色都会变成一个 **worker group 门面对象**。控制器不去想“第 2 个节点第 7 张卡上的 rank 在干嘛”，它只想“actor_rollout_wg 现在该执行哪件事”。

### 第三遍：读 `fit()`

`fit()` 是整个训练逻辑的主轴，值得细读。

## 4. `fit()` 这条主线，用人话怎么讲？

```{mermaid}
flowchart TD
    A[从 dataloader 拿到 batch_dict] --> B[封装成 DataProto]
    B --> C[挂 uid 与 meta_info]
    C --> D[构造生成用 batch]
    D --> E[生成 response]
    E --> F[并回生成结果]
    F --> G[计算 reward 与补充信息]
    G --> H[重算或旁路 old_log_probs]
    H --> I[可选：reference log-probs]
    I --> J[可选：critic values]
    J --> K[KL / rollout correction]
    K --> L[控制器上算 advantage]
    L --> M[更新 critic]
    L --> N[更新 actor]
    M --> O[validation / metrics / checkpoint]
    N --> O
```

下面按顺序拆开。

### 步骤 A：`batch_dict -> DataProto`

dataloader 先给你一个普通字典，控制器马上把它包成 `DataProto`。

这一步非常关键，因为从这里开始，后面每个阶段都能往同一个容器里继续追加字段，而不是重新约定一套输入输出结构。

### 步骤 B：构造生成 batch 与重复采样

`_get_gen_batch(...)` 会抽出 rollout 真正需要的部分。

如果：

- `rollout.n > 1`

那么同一个 prompt 还会被重复，以便做 grouped sampling。GRPO 场景里这一步尤其重要。

### 步骤 C：rollout

控制器会调用：

`self.async_rollout_manager.generate_sequences(...)`

从这一刻开始，你就真正离开“舒服的单机 Python 主循环”了，进入昂贵的分布式推理阶段。

底层可能是：

- vLLM
- SGLang
- 或其他 rollout backend

它们都在 `verl/workers/rollout/*` 下面。

### 步骤 D：reward

生成结束后，reward 往往分成两层：

1. 可选的 reward model 分数
2. 规则式 / 自定义 reward 抽取

这样拆是为了组合灵活：

- 数学题、代码题常常喜欢规则奖励
- 偏好学习更常见 reward model
- 更复杂场景会把两者混起来

### 步骤 E：old log-prob 与 rollout log-prob

这是很多人第一次读 `ray_trainer.py` 时最容易忽略，但实际上非常关键的一段。

`verl` 可以跑两种模式：

- **bypass mode**：直接复用 rollout log-probs 作为近端锚点
- **decoupled mode**：单独重算 `old_log_probs`，进入三策略视角

这就是它从“普通 PPO”跨到“支持 rollout correction 的 PPO 变体”的关键桥梁。

### 步骤 F：reference 与 critic

如果当前配置启用，trainer 还会单独算：

- reference log-probs
- critic values

这也是为什么 GRPO 通常比 PPO 便宜：很多时候它根本就不需要 critic 这条路。

### 步骤 G：KL、rollout correction 与 advantage

控制器接着会做三件事：

- 可选地把 KL 作为 reward penalty 加进去
- 可选地算 rollout correction 权重和相关指标
- 在控制器本地计算 advantage

这里的设计很有意思：advantage 计算相对轻，没必要为了它再拉起一套重的分布式模型图。

### 步骤 H：actor / critic update

直到前面的数据都齐了，控制器才会真正发起重计算：

- `self._update_critic(batch)`
- `self._update_actor(batch)`

这两次调用才会进入后端 worker 的重型优化逻辑。

## 5. 为什么 `@register` 是隐藏 MVP？

如果你想明白“为什么控制器一句话能正确驱动很多远端 worker 干活”，必读：

- `verl/single_controller/base/decorator.py`
- `verl/single_controller/base/worker_group.py`
- `docs/single_controller.rst`

核心机制非常清楚：

- worker 方法先带上 dispatch 元信息
- `WorkerGroup` 绑定方法时读取这些元信息
- 控制器的普通调用被翻译成拓扑感知的分布式 RPC

比如 `generate_sequences(...)` 可能按 DP 方式切数据，而 Megatron 的更新路径会用更复杂的 Megatron-aware dispatch。

## 6. `DataProto` 需要你脑中有一张图

很多人把 `DataProto` 当作小细节，但实际上它就是全流程的通用语言。

```{mermaid}
flowchart LR
    A[Prompts] --> B[DataProto]
    C[Responses] --> B
    D[old_log_probs] --> B
    E[ref_log_prob] --> B
    F[values] --> B
    G[token_level_rewards] --> B
    H[advantages / returns] --> B
    B --> I[actor update]
    B --> J[critic update]
```

你调试时一旦搞清楚“当前 `DataProto` 里应该有哪些字段”，很多奇怪问题会立刻变得可解释。

## 7. 真正的后端分叉从 worker 文件开始

当你已经看懂 trainer 主循环，再去看后端 worker。

### FSDP 路径

先看：

- `verl/workers/fsdp_workers.py`
- `docs/workers/fsdp_workers.rst`

优先盯住这些方法：

- `init_model`
- `generate_sequences`
- `compute_ref_log_prob`
- `compute_values`
- `update_actor`
- `update_critic`
- `compute_rm_score`

FSDP 路径的心智负担通常更低，因为模型定义离 Hugging Face 更近。

### Megatron 路径

再看：

- `verl/workers/megatron_workers.py`
- `docs/workers/megatron_workers.rst`

方法名大体类似，但 dispatch mode 会明显更复杂，因为它要照顾 TP / PP / DP / CP / EP 等并行结构。

也正因为此，`verl` 的控制器抽象才显得尤其珍贵：外层流程不怎么变，底层并行组织却能复杂很多。

## 8. 我最推荐的源码阅读顺序

如果你想用一条线把仓库主轴串起来，建议按这个顺序读：

1. `verl/trainer/main_ppo.py`
2. `verl/trainer/ppo/ray_trainer.py`
3. `verl/protocol.py`
4. `verl/single_controller/base/decorator.py`
5. `verl/single_controller/base/worker_group.py`
6. `verl/single_controller/ray/base.py`
7. `verl/workers/fsdp_workers.py`
8. `verl/workers/megatron_workers.py`
9. `verl/workers/rollout/base.py`
10. 任意一个具体 rollout backend：`verl/workers/rollout/*`
11. `verl/trainer/ppo/core_algos.py`
12. reward 相关文件：`verl/trainer/ppo/reward.py` 与 `verl/workers/reward_manager/*`

这个顺序的好处在于：

- 先看谁在指挥
- 再看调用如何绑定
- 再看底层后端到底算什么
- 最后看数学是如何落到代码上的

## 9. 真出问题了，该先 debug 哪里？

| 症状 | 最值得先看的地方 |
| --- | --- |
| response 一生成就不对 | rollout backend 文件 + 采样配置 |
| KL 指标很怪 | `ray_trainer.py` 里的 `apply_kl_penalty`，以及 `core_algos.py` 的 KL 逻辑 |
| GRPO 分组怪怪的 | `compute_grpo_outcome_advantage`，重点看 `uid` 分组 |
| actor 更新发散 | worker 里的 `update_actor`、policy loss 配置、rollout correction 配置 |
| reward 时灵时不灵 | `extract_reward`、reward manager、自定义 reward function 路径 |
| rank / shard / 拓扑看不懂 | `decorator.py`、`worker_group.py`、Megatron dispatch |

## 10. 给贡献者的最后一句建议

你想扩展 `verl` 时，不要一上来先改最底层 worker。

先按顺序问自己三个问题：

1. **我改的是控制器主循环的哪个阶段？**
2. **我需要往 `DataProto` 里新增哪些字段？**
3. **真正的重计算应该落在哪个 worker 方法 / dispatch mode 上？**

只要你保持这个阅读和改造顺序，就不容易把算法逻辑和后端实现越写越耦合。

下一篇建议读 [`math-theory.md`](./math-theory.md)，把这些代码路径和底层数学对应起来。
