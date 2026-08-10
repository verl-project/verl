# SAPO × Megatron × NPU 阶段性运行报告

Last updated: 08/09/2026.

**汇报日期**：2026-08-09 ｜ **任务**：`682143`（PYTORCHJOB，2 节点 × 8 NPU = 16 × 910B）

---

## 一、任务概述

在昇腾 910B 集群上打通 **Qwen3-30B-A3B（MoE）+ SAPO 算法 + Megatron 后端 + vllm_ascend 推理** 的完整 RL 训练链路，跑 100 步验证训练可用性与收敛。

| 项 | 配置 |
|---|---|
| 模型 | Qwen3-30B-A3B（128 experts），预转 mcore dist checkpoint |
| 并行 | TP=4 / EP=4 / ETP=4 / PP=1，DP=4（16 rank 全 MoE 专家分片）|
| 后端 | Megatron 训练 + vLLM（vllm_ascend）rollout，`model_engine=megatron` |
| 算法 | SAPO（`policy_loss.loss_mode=sapo`），`use_kl_loss=False`，GRPO 采样 |
| 数据 | dapo-math-17k 训练集，AIME-2024 评测集 |
| 关键配置 | 全 offload（param/optimizer/grad）、`RECOMPUTE=full`、`micro_batch=1`、`SAVE_FREQ=100`（末次存档）、checkpoint 写节点本地盘 |

---

## 二、机器配置

### 2.1 硬件

| 项 | 配置 |
|---|---|
| 集群 | 昇腾 910B（wulan），资源池 infra910b |
| 节点 | 2 节点，每节点 CPU **192 核**（Kunpeng-920，aarch64） |
| 加速卡 | **910B3 × 16**（每节点 8 卡），每卡 HBM **60.96 GiB** |
| 网络 | 节点间 NPU-HCCS 直连（HybridCube，Tier-2/3） |
| 存储 | JuiceFS 共享文件系统 + 节点本地 overlay（checkpoint 写本地盘） |

### 2.2 软件栈

| 软件 | 版本 |
|---|---|
| Python | 3.11.15 |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post2 |
| CANN | ascend-toolkit 9.0.0 |
| megatron-core | 0.16.2 |
| vllm / vllm-ascend | 0.18.0+empty / 0.18.1.dev41 |
| transformers | 5.3.0.dev0 |
| flash-linear-attention | 0.5.0 |

### 2.3 数据与模型规模

| 数据/模型 | 规模 |
|---|---|
| 训练集 dapo-math-17k/train.parquet | **1,791,700 条**（286 MB）|
| 评测集 AIME-2024/test.parquet | 960 条 |
| 预转 Megatron dist checkpoint（Qwen3-30B-A3B-Base-mcore）| 57 GB |

---

## 三、训练配置（任务 682143 实际生效值）

| 类别 | 参数 | 值 | 说明 |
|---|---|---|---|
| 并行 | TP / PP / CP / EP / ETP | 4 / 1 / 1 / 4 / 4 | MoE：EP×ETP = 16 = world_size |
| | GEN_TP | 4 | rollout 生成 TP |
| 训练批次 | TRAIN_BATCH_SIZE | 96 | 全局 batch |
| | PPO_MINI_BATCH_SIZE | 32 | 每优化步 mini-batch |
| | PPO_MICRO_BATCH_SIZE_PER_GPU | **1** | 吞吐瓶颈（见 4.3）|
| 长度 | MAX_PROMPT_LENGTH | 2048 | |
| | MAX_RESPONSE_LENGTH | 4096 | |
| | PPO_MAX_TOKEN_LEN_PER_GPU | 8192 | offload 下压小以避免 OOM |
| SAPO | TAU_POS / TAU_NEG | 1.0 / 1.05 | 平滑温度（论文默认）|
| | loss_agg_mode | seq-mean-token-mean | 硬编码 |
| | use_kl_loss / entropy_coeff | False / 0 | SAPO 无 KL |
| 优化器 | ACTOR_LR | 1e-6 | |
| | optimizer_cpu_offload + fraction | True + 1 | HybridDeviceOptimizer |
| 内存 | megatron.param/optimizer/grad offload | 全 True | 额外一层（本轮 probe 证与吞吐无关）|
| | RECOMPUTE | full | 每层重算，最省显存 |
| rollout | ROLLOUT_N / ROLLOUT_GPU_MEM_UTIL | 8 / 0.8 | 每组 8 个 response |
| 存档 | SAVE_FREQ / MAX_ACTOR_CKPT_TO_KEEP | 100 / 1 | 末次存档 |
| | default_local_dir | 节点本地盘 | 见失败根因 |
| 步数 | total_training_steps / TOTAL_EPOCHS | 100 / 1 | |

---

## 四、运行结果

### 4.1 训练完成度

**99/100 步跑完**，训练循环耗时约 **34.6 小时**（step 1 起），全程无 OOM、无算子错误、无通信异常。训练本身是**跑通的**。

### 4.2 Reward 收敛曲线（critic/rewards/mean，每 20 步采样）

| step | ~20 | ~40 | ~60 | ~80 | ~99 |
|---|---|---|---|---|---|
| reward | **-0.862** | -0.536 | -0.292 | -0.227 | **~-0.05** |

全程**单调上升**（-0.86 → -0.05），后期在 0 附近小幅波动（末期出现过 +0.16 的样本），符合 GRPO 后期探索特征。无发散、无 KL 爆表。

### 4.3 性能指标

| 指标 | 早期 | 稳态（step 40+）|
|---|---|---|
| step 耗时 | 1481 s | **~1260 s**（约 21 分钟/步）|
| throughput（token/卡/s）| 32 | **~52–56**，全程缓升 |
| update_actor 占比 | — | **~777–791 s，占 step 的 62%** |
| 显存峰值 | — | **15.8 / 60.96 GiB**（仅 1/4，余量充足）|

> 验收口径 `perf/throughput > 100` 目前约 **一半**（52–56），瓶颈集中在 `update_actor`（训练前反向），与 rollout 无关。
>
> **micro_batch=2 探针（692443）回报**：`update_actor` 775s → **458–464 s（约 -40%）**，显存 15.8→16.1 GiB，step 1260→~970 s——印证 `micro_batch=1` 是吞吐瓶颈，下一轮将采用该配置。

### 4.4 算子级耗时（actor_update，torch.op_mark 离线分析）

对 `actor_update` 窗口（step 3 训练更新，25 分钟）的 `FRAMEWORK/torch.op_mark` TLV 事件做 enqueue/dequeue 配对聚合，得到每算子设备执行时长。**HCCL 通信占设备执行总时长 56%**：

**DEQUEUE（设备执行，总计 42.1 s）Top 15：**

| op | total_s | count | avg_ms | 类别 |
|---|---|---|---|---|
| **HcclAlltoAllV** | **16.1** | 82,944 | 0.19 | 通信（MoE 专家分发）|
| aclnnInplaceCopy | 3.5 | 1,361,619 | 0.00 | 拷贝 |
| HcclAllGather | 3.1 | 55,878 | 0.06 | 通信 |
| HcclAllGatherV | 1.8 | 46,080 | 0.04 | 通信 |
| HcclReduceScatterV | 1.5 | 36,864 | 0.04 | 通信 |
| record_event | 1.4 | 992,067 | 0.00 | 同步 |
| aclnnCat | 1.3 | 230,978 | 0.01 | 内存拼接 |
| HcclReduceScatter | 1.1 | 28,038 | 0.04 | 通信 |
| aclnnMul | 1.1 | 224,441 | 0.00 | 逐元素 |
| aclnnInplaceAdd | 1.1 | 692,004 | 0.00 | 逐元素 |
| aclnnFlashAttentionVarLenScore | 0.8 | 18,432 | 0.04 | 注意力 |
| wait_event | 0.6 | 566,148 | 0.00 | 同步 |
| aclnnGroupedMatmulV5 | 0.6 | 73,728 | 0.01 | 计算（MoE FFN）|
| aclnnRmsNorm | 0.4 | 73,920 | 0.01 | 归一化 |
| aclnnMm | 0.4 | 36,864 | 0.01 | 计算 |

**ENQUEUE（host 侧 launch，总计 3.4 s）**——主机发射开销极小，瓶颈全在设备端：

| op | total_s | count | 说明 |
|---|---|---|---|
| aclnnInplaceCopy | 0.8 | 1,361,619 | 拷贝 |
| record_event | 0.4 | 992,067 | 同步 |
| aclnnInplaceAdd | 0.4 | 692,004 | 逐元素 |
| …其余均 <5% | | | |

> **解读**：
> 1. **HCCL 通信合计 ~23.6 s、占设备执行 56%** —— `HcclAlltoAllV` 单项 38%，是 **MoE 专家分发**（128 expert × EP=4，逐层逐 token 组 all-to-all）；`AllGather/ReduceScatter` 是 DP 梯度规约。
> 2. **计算算子单次 <0.1 ms、合计 <6%**（FlashAttention 0.8s、GroupedMatmul 0.6s、Mm 0.4s）——"算得慢"被彻底排除。
> 3. 这解释了 micro_batch=2 的 -40%：更大的 micro-batch 让 **AlltoAllV/AllGather 的同步间隙可被计算重叠**；也解释了 offload 无效（瓶颈不在搬运）。
> 4. 下一个吞吐 lever 是**压 MoE 通信**（EP 布局 / all-to-all 分段 / 计算-通信 overlap）。

---

## 五、失败与根因

任务最终状态 **Failed**：**99 步训练全部成功，死在 step 100 的最后一次 checkpoint 保存**。

**时序**：save 开始 08-08 23:41 → 30 分钟 gloo barrier 超时 → 08-09 00:58 崩溃退出。

**根因（已在节点上核对文件实物，非推断）**：

- **node1（rank 8–15）**：本地盘 **~2 分钟**写完 8 个分片，镜像到 JuiceFS 完整；
- **node0（rank 0–7）**：**76 分钟零分片写出**，`.metadata`（collective 产物）缺失；
- 全体在 `torch.distributed.barrier()` 上等 node0，满 30 分钟超时后崩溃。

**结论**：checkpoint 不可用（缺 node0 半份 + `.metadata`）。这是一个**结构性阻塞**——只要 node0 在存档时卡住，任何跑到终点的 100 步 run 都会死。已排除存储层（node1 本地盘 2 分钟证明很快）、已排除"单 rank 慢"（是 **node0 整节点 blocked**）。

---

## 六、关键发现与下一步

### 6.1 已确认的结论

1. **训练链路打通**：SAPO + Megatron + vllm_ascend 在 16 × 910B3 上可稳定训练 99 步，reward 收敛。
2. **update_actor 是吞吐瓶颈**（62%）：算子级分析显示 **HCCL 通信占设备执行 56%**（MoE AlltoAllV 单项 38%），计算算子合计 <6%——不是算得慢，是通信同步间隙。
3. **offload 假设被证伪**：4 步 probe 关掉 verl 层全 offload，`update_actor` 分毫未降（781s），显存占用不变——瓶颈不在 offload 搬运。
4. **checkpoint 落盘 node0 卡死**：独立于存储，需专项定位。

### 6.2 进行中的探针（并行，不占长任务）

| 探针 | 目的 | 状态 |
|---|---|---|
| **692443** micro2 | `micro_batch=2` 验证 update_actor 是否下降 | 已完成（update_actor 458s，-40%，结论纳入 4.3）|
| **692651** node0-ckpt | 本地盘复现 node0 卡死 + py-spy 抓栈 | 已复现，采样数据落 JuiceFS，待 stack 分析 |

### 6.3 下一步规划

1. **吞吐**：micro_batch=2 已 -40%，下一轮 100 步直接采用（可再探 4）；
2. **通信**：基于算子级结论（MoE AlltoAllV 38%），研究 EP 布局 / all-to-all 分段 / 计算-通信 overlap 以进一步压通信；
3. **checkpoint**：node0 卡死根因定位后重跑，确保 checkpoint 完整可用（评测依赖）；
4. 目标：`throughput > 70`（micro=2 后）+ reward 收敛 + 可评测 checkpoint 三线齐备。

---

*注：本报告基于任务 682143 运行日志、节点文件实物核对及探针实验整理。*
