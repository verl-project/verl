# 快速上手：第一次跑通 `verl` 的 PPO / GRPO

这篇不是简单把官方 quickstart 翻译一遍，而是想解决一个更现实的问题：

> 命令能跑，不代表你知道系统在干什么。

所以我们会一边给出最小路径，一边解释每个阶段的系统意义、源码落点以及日志该怎么看。

## 1. 为什么 `verl` 的使用感和很多训练框架不一样？

很多 RL 训练代码把控制逻辑直接埋进一个庞大的分布式脚本里。`verl` 不这么做。它故意把主流程摆在明面上：

1. 读一批 prompt
2. 生成 response
3. 计算 reward
4. 计算 advantage
5. 更新 actor / critic

听起来像五步小菜谱，但每一步底下都可能是很多 GPU、很多进程、甚至不同后端引擎在协同工作。

```{mermaid}
flowchart LR
    A[Parquet 数据集] --> B[RLHFDataset<br/>+ dataloader]
    B --> C[控制器封装 DataProto]
    C --> D[Rollout 引擎<br/>vLLM / SGLang / HF]
    D --> E[Reward 函数<br/>或 Reward Model]
    E --> F[控制器上算 Advantage]
    F --> G[Critic 更新]
    F --> H[Actor 更新]
    G --> I[Checkpoint / 验证 / 日志]
    H --> I
```

## 2. 最小可用起步姿势

第一次上手，建议走最朴素的路径：

- 把 GSM8K 一类数据先预处理成 parquet
- 选一个体量小、好放下的 Hugging Face 模型
- 用 `verl.trainer.main_ppo` 起一个单节点、单 GPU 的 PPO 跑通
- 先理解 FSDP 风格路径，再考虑 Megatron 和复杂拓扑

最关键的源码入口有四个：

- 数据预处理：`examples/data_preprocess/*`
- 官方 quickstart：`docs/start/quickstart.rst`
- PPO 入口：`verl/trainer/main_ppo.py`
- PPO 主循环：`verl/trainer/ppo/ray_trainer.py`

## 3. 数据准备：为什么先转 parquet？

官方 quickstart 用的是：

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

为什么不直接每轮读原始 JSON？

- reward 相关字段可以提前标准化
- 训练期读取更快、更稳定
- 主循环不用每轮重复做重文本预处理

你可以把这一步想成开饭店前先把食材分门别类装进冷柜，而不是每来一桌客人都去拆一大袋批发菜。

## 4. 最小 PPO 命令怎么理解？

一个代表性的单卡命令是：

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  critic.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  critic.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  trainer.logger=console \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=15
```

这条命令不是“能跑就行”的咒语，而是一整套系统契约：

- `verl.trainer.main_ppo` 决定顶层入口和 Hydra 配置装配
- `actor_rollout_ref.*` 决定 actor、rollout、reference 这一族模块怎么组织
- `critic.*` 决定 value model 的路径与更新参数
- `trainer.*` 决定整个任务的调度、日志、节点规模

## 5. 这条命令背后会唤醒哪些文件？

```{mermaid}
flowchart TD
    A[`python -m verl.trainer.main_ppo`] --> B[`main_ppo.py`<br/>Hydra + Ray 初始化]
    B --> C[`TaskRunner`<br/>决定 worker 与资源池]
    C --> D[`RayPPOTrainer`<br/>初始化 dataloader 与 worker group]
    D --> E[`fit()`<br/>rollout -> reward -> advantage -> update]
    E --> F[`fsdp_workers.py` 或 `megatron_workers.py`]
    E --> G[`core_algos.py`]
```

这也是为什么我建议你第一次读源码时，一定从 `main_ppo.py` 开始，不要一上来就扎进最底层 worker。

## 6. 训练日志应该怎么读？

常见指标包括：

- `timing/gen`
- `timing/ref`
- `timing/values`
- `timing/update_actor`
- `timing/update_critic`
- `actor/pg_loss`
- `critic/vf_loss`
- `response_length/mean`
- `val/test_score/...`

这些指标的现实含义如下：

| 指标 | 该怎么理解 |
| --- | --- |
| `timing/gen` | rollout 生成阶段花了多久。它大，通常先看 rollout 后端、TP 配置和显存利用率。 |
| `timing/ref` | reference policy 计算 log-prob 的成本。开了 KL 或某些 GRPO 配置时会更敏感。 |
| `timing/values` | critic 前向计算耗时。PPO 很关键，GRPO 往往没有这部分。 |
| `timing/update_actor` | actor 优化阶段耗时，通常和 micro-batch、序列长度关系很大。 |
| `timing/update_critic` | critic 优化阶段耗时，用来判断 value 路径是否卡住。 |
| `actor/pg_loss` | 策略梯度目标本身。不同 loss 聚合方式下，绝对值不宜横向乱比。 |
| `critic/vf_loss` | value function loss。如果它爆炸，优势估计常常也会变得很 noisy。 |
| `response_length/mean` | 一眼看出 rollout 成本是否在持续变重。response 越长，生成和更新都越贵。 |

## 7. 新手最该先调哪些参数？

建议按这个顺序理解和调：

1. **模型路径** -- 先保证能装得下。
2. **rollout 后端** -- 常见默认是 `vllm`，也可以看 `sglang`、TRT-LLM。
3. **batch / micro-batch** -- 这是显存和吞吐的第一开关。
4. **response length** -- 输出一长，rollout 和 update 都跟着变贵。
5. **placement** -- 先 colocated，搞清楚之后再玩 split placement。

## 8. 什么时候应该切到 GRPO？

如果你希望：

- 保留传统 actor-critic 路线
- 显式训练 value model
- 使用 GAE 一类优势估计

那就先用 PPO。

如果你更在意：

- 不想训练 critic
- 一个 prompt 多采样几个答案做组内相对比较
- 想直接走 grouped rollouts 的路径

那就更适合 GRPO。

在 `verl` 里，切换的关键点主要有两个：

- 把 `algorithm.adv_estimator` 设成 `grpo`
- 把 `actor_rollout_ref.rollout.n` 设成大于 1

源码与文档锚点：

- 算法文档：`docs/algo/ppo.md`、`docs/algo/grpo.md`
- 代码实现：`verl/trainer/ppo/core_algos.py`

## 9. FSDP 和 Megatron 先怎么选？

| 后端 | 什么时候优先用 | 代价是什么 |
| --- | --- | --- |
| FSDP | 做算法原型、加新 Hugging Face 模型、想先把复杂度压下来时 | 好扩展，但极限扩展性与重分片效率通常不如 Megatron |
| Megatron | 追求大规模并行、极限吞吐、超大模型训练时 | 并行概念更多，理解与维护门槛更高 |

## 10. 下一步看什么？

- 想先看全局系统骨架：读 [`architecture.md`](./architecture.md)
- 想顺着代码一路看下去：读 [`source-code-tour.md`](./source-code-tour.md)
- 想把公式彻底看明白：读 [`math-theory.md`](./math-theory.md)

同时建议把这些官方文档一起打开：

- `docs/start/install.rst`
- `docs/start/quickstart.rst`
- `docs/examples/ppo_code_architecture.rst`
- `examples/ppo_trainer/README.md`
- `examples/grpo_trainer/README.md`
