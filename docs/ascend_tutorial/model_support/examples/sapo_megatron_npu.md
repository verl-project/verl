# SAPO × Megatron × NPU 运行报告

Last updated: 08/15/2026.

**汇报日期**：2026-08-15 ｜ **验收任务**：`716535`（100 步验收 run，Succeeded）｜ **早期失败任务**：`682143`（682143 死在 step 100 checkpoint 落盘，见 §五）

> **状态结论（TL;DR）**：SAPO×Megatron×NPU 训练链路**已打通并通过验收**。100 步 run（task 716535，probe6 配置）`throughput` 均值 **123.94**（50 读数全 >100），reward 前25均值 -0.1474→后24 -0.0496（+66% 向零），双节点 exit 0，无 OOM/AssertionError/gloo timeout。checkpoint 落盘隐患仍未修（本轮 `SAVE_FREQ=-1` 规避），详见 §6.4。

---

## 一、任务概述

在昇腾 910B 集群上打通 **Qwen3-30B-A3B（MoE）+ SAPO 算法 + Megatron 后端 + vllm_ascend 推理** 的完整 RL 训练链路，跑 100 步验证训练可用性与收敛。

| 项 | 682143（早期失败 run） | 716535（验收 run，probe6）|
|---|---|---|
| 模型 | Qwen3-30B-A3B（128 experts），预转 mcore dist checkpoint | 同左 |
| 并行 | TP=4 / EP=4 / ETP=4 / PP=1，DP=4 | **TP=4 / EP=8 / ETP=1** / PP=1，DP=4（EP8 压通信）|
| 后端 | Megatron 训练 + vLLM（vllm_ascend）rollout | 同左 |
| 算法 | SAPO（`policy_loss.loss_mode=sapo`），`use_kl_loss=False`，GRPO 采样 | 同左 |
| 数据 | dapo-math-17k 训练集，AIME-2024 评测集 | 同左 |
| 关键配置 | 全 offload、`RECOMPUTE=full`、`micro_batch=1`、`SAVE_FREQ=100` | full offload、`RECOMPUTE=full`、**`micro_batch=4`**（打包上限）、**三标志 dynamic_bsz**、`SAVE_FREQ=-1`（规避 checkpoint 隐患）、`PROFILE=0` |
| 结果 | 99 步训练成功，死在 step 100 checkpoint 落盘（Failed）| **100 步 Succeeded，throughput 123.94 >100 ✓，reward +66% ✓** |

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

## 三、训练配置

下表左列为 682143（早期失败 run）实际生效值，右列为 716535（验收 run，probe6 配置）实际生效值。验收 run 的配置即 recipe 基线（见 §6.3）。

| 类别 | 参数 | 682143 | 716535（probe6，验收基线）| 说明 |
|---|---|---|---|---|
| 并行 | TP / PP / CP / EP / ETP | 4 / 1 / 1 / 4 / 4 | 4 / 1 / 1 / **8 / 1** | EP8 压通信：AlltoAllV 绝对时长 -70%（见 4.4）|
| | GEN_TP | 4 | 4 | rollout 生成 TP |
| 训练批次 | TRAIN_BATCH_SIZE | 96 | 96 | 全局 batch |
| | PPO_MINI_BATCH_SIZE | 32 | 32 | 每优化步 mini-batch |
| | PPO_MICRO_BATCH_SIZE_PER_GPU | 1 | **4** | 打包上限（dynamic_bsz 装箱，见下）|
| 长度 | MAX_PROMPT_LENGTH | 2048 | 2048 | |
| | MAX_RESPONSE_LENGTH | 4096 | 4096 | |
| | PPO_MAX_TOKEN_LEN_PER_GPU | 8192 | 8192 | token 预算（dynamic_bsz 装箱上限）|
| dynamic_bsz | actor.use_dynamic_bsz | False | **True** | throughput 破 100 关键（见 4.3）|
| | rollout.log_prob_use_dynamic_bsz | False | **True** | 三标志必须同开（`engine_workers.py:561` 双向断言）|
| | ref.log_prob_use_dynamic_bsz | False | **True** | ref 静默不校验但必须一致 |
| SAPO | TAU_POS / TAU_NEG | 1.0 / 1.05 | 1.0 / 1.05 | 平滑温度（论文默认）|
| | loss_agg_mode | seq-mean-token-mean | seq-mean-token-mean | 硬编码 |
| | use_kl_loss / entropy_coeff | False / 0 | False / 0 | SAPO 无 KL |
| 优化器 | ACTOR_LR | 1e-6 | 1e-6 | |
| | optimizer_cpu_offload + fraction | True + 1 | True + 1 | HybridDeviceOptimizer（不可关，707032 证实关则 Adam 态上设备 OOM）|
| 内存 | megatron.param/optimizer/grad offload | 全 True | param/grad True，**optimizer False** | 707489 证实 verl 层 optimizer_offload 冗余 |
| | RECOMPUTE | full | full | 每层重算，最省显存（probe3 证实 selective 仅 -10.4% 不值得）|
| rollout | ROLLOUT_N / ROLLOUT_GPU_MEM_UTIL | 8 / 0.8 | 8 / **0.6** | probe6 实测值 |
| 存档 | SAVE_FREQ / MAX_ACTOR_CKPT_TO_KEEP | 100 / 1 | **-1** / 1 | 规避 sync barrier 崩溃面（见 §6.4）|
| | default_local_dir | 节点本地盘 | — | 验收 run 不落盘 |
| 步数 | total_training_steps / TOTAL_EPOCHS | 100 / 1 | 100 / 1 | |
| 调试 | PROFILE | 0 | **0** | 避免 3.3× 污染（707489 实测）|
| | EXPERIMENT_NAME | — | qwen3_30b_a3b_megatron_16npu_100step_probe6 | resume_mode=auto + 同名会捡旧 ckpt，必须换名 |

---

## 四、运行结果

### 4.0 验收结论（task 716535，2026-08-13，Succeeded）

100 步验收 run（probe6 配置）**Succeeded**，约 **12.8 小时**（07:51:55→20:52:05），stopReason None，node0/node1 双节点 `exit code: 0`。**两项验收判据均通过**：

| 验收判据 | 口径 | 结果 |
|---|---|---|
| `perf/throughput > 100` | `total_num_tokens/(time*n_gpus)`（`metric_utils.py:669`，每卡每秒 token 数）| **✓ 通过**：50 读数全 >100，均值 123.94 |
| reward 上升 | `critic/rewards/mean` 趋势 | **✓ 通过**：前25 -0.1474→后24 -0.0496（+66% 向零）|

> 日志本地转储 `/tmp/716535_full.log`（252941 字节，`cctl pytorchjob logs 716535 --pod all --no-input` 获取）。日志捕获 step 44-100（49 步，1-43 不在存档窗口）。

### 4.1 训练完成度（716535）

**100/100 步跑完**，训练循环约 **12.8 小时**，全程无 OOM、无算子错误、无通信异常、无 AssertionError、无 gloo timeout。step 100 正常到 `is_last_step` return（非 barrier 超时崩溃）。`max_memory_allocated_gb` 跨全程恒定 **29.22 GiB**（与 probe6 step4 一致，无泄漏）。teardown 方案 A 正常：node1 检测 head 死亡后 `ray head gone, exiting` 干净退出，无 GCS 僵尸。

### 4.2 Reward 收敛曲线（716535，critic/rewards/mean）

`critic/rewards/mean` 围零振荡（GRPO 探索，与 682143 基线同构），全程**向零收敛**：

| 区间 | 前25均值 | 后24均值 | min | max |
|---|---|---|---|---|
| reward | **-0.1474** | **-0.0496** | -0.365（step48）| **+0.148**（step100，收尾最高）|

**+66% 向零**，收尾 step100 达全程最高 +0.148，符合 GRPO 后期探索特征。无发散、无 KL 爆表。

### 4.3 性能指标（716535，step 44-100）

| 指标 | 值 | 说明 |
|---|---|---|
| throughput（token/卡/s）| 均值 **123.94**，min 110.98（step44）/ max 147.16（step100）| **50 读数全 >100 ✓** |
| throughput 趋势 | 前25均值 119.06 → 后25均值 128.83 | **+8.2% 缓升** |
| step 耗时 | ~450 s/步（无 profile 污染）| probe6 step4 clean 449.1s |
| 显存峰值 | **29.22 / 60.96 GiB**（48%）| 恒定，无泄漏 |
| update_actor（probe6 step4 clean）| 180.8 s | 见 4.4，dynamic_bsz + EP8 压降 |

> **对比 682143 早期 run**：throughput 52-56（约一半）→ 123.94（**+125%**），update_actor 777-791s → 180.8s（**-77%**），显存 15.8→29.22 GiB（吃满更多但仍 48%）。提升来自三处叠加：(1) EP8 压通信（AlltoAllV -70%）；(2) micro_batch=4 打包上限（launch 次数减半）；(3) **三标志 dynamic_bsz**（token 预算装箱，throughput 破 100 的关键 lever）。

### 4.4 算子级耗时与优化路径（actor_update，torch.op_mark 离线分析 + probe 实测）

**682143 基线算子级 profile**（step 3 训练更新窗口，TP4/EP4/ETP4）——**HCCL 通信占设备执行总时长 56%**：

**DEQUEUE（设备执行，总计 42.1 s）Top：**

| op | total_s | count | avg_ms | 类别 |
|---|---|---|---|---|
| **HcclAlltoAllV** | **16.1** | 82,944 | 0.19 | 通信（MoE 专家分发）|
| aclnnInplaceCopy | 3.5 | 1,361,619 | 0.00 | 拷贝 |
| HcclAllGather | 3.1 | 55,878 | 0.06 | 通信 |
| HcclAllGatherV | 1.8 | 46,080 | 0.04 | 通信 |
| HcclReduceScatterV | 1.5 | 36,864 | 0.04 | 通信 |
| …计算算子合计 <6% | | | | |

> **解读**：HCCL 通信合计 ~23.6 s、占设备执行 56%，`HcclAlltoAllV` 单项 38% 是 MoE 专家分发（128 expert × EP=4，逐层逐 token 组 all-to-all）；计算算子单次 <0.1 ms、合计 <6%——"算得慢"被排除。

**probe 优化路径（2026-08-11→13，逐步定位 throughput 瓶颈）**：

| probe | toggle | update_actor（step4 clean）| throughput | 判定 |
|---|---|---|---|---|
| 707489 基线 | TP4/EP8/ETP1，full+micro2，profile-on | 360.1 s | 64.0 | 基线（profile 污染 +65-68%）|
| 703719 | EP4/EDP4 | — | — | 未达预期，EP8 更优 |
| probe3 (710229) | RECOMPUTE full→selective | 322.7 s（-10.4%）| 63.0 | **行3：recompute 非主因**（未达 >20% 阈值，且多用 7.6G 显存）|
| probe4 (711571) | micro2→**4** | 256.8 s（**-28.7%**）| 72.6（+13.4%）| **行1命中**：micro launch overhead 是真实瓶颈 |
| probe5 (712691) | micro4→**8** | 189.8 s（-26.1%）| 80.9（+11.4%）| **行1命中**：micro8 是静态最优，但 throughput 未破 100 |
| probe6 (712804) | +**三标志 dynamic_bsz** | 180.8 s（-4.7%）| **113.2（+40.0%）** | **行1命中**：dynamic_bsz 胜出，throughput 首次破 100 ✓✓ |

**probe6 dynamic_bsz 机制**：`use_dynamic_bsz=True` 时 `rearrange_micro_batches`（`verl/workers/engine/utils.py:73-94`）按 token 预算（8192）装箱而非固定序列数切分。最大增量在 `old_log_prob`：micro8 173.9s → dynamic 30.5s（-82.5%，log_prob 纯前向，token 预算装箱能更激进塞满）。`global_seqlen/balanced_min/max` 证实装箱后 rank 间实际计算 token 量均衡（差仅 30 token），这是 micro8 固定切分做不到的。

**最终生产配置锁定**：full + micro4（打包上限）+ 8192（token 预算）+ 三标志 dynamic_bsz。100 步验收 run（716535）即此配置，throughput 持续 >100 + reward 上升，训练链路打通。

---

## 五、失败与根因（历史：682143 早期 run）

> 本节记录 682143 早期 run 的失败，**已在 716535 验收 run 中通过 `SAVE_FREQ=-1` 规避**（不落盘，避开 barrier 面）。checkpoint 落盘隐患的根因与修复方案见 §6.4。

任务 682143 最终状态 **Failed**：**99 步训练全部成功，死在 step 100 的最后一次 checkpoint 保存**。

**时序**：save 开始 08-08 23:41 → 30 分钟 gloo barrier 超时 → 08-09 00:58 崩溃退出。

**根因（已在节点上核对文件实物，非推断）**：

- **node1（rank 8–15）**：本地盘 **~2 分钟**写完 8 个分片，镜像到 JuiceFS 完整；
- **node0（rank 0–7）**：**76 分钟零分片写出**，`.metadata`（collective 产物）缺失；
- 全体在 `torch.distributed.barrier()` 上等 node0，满 30 分钟超时后崩溃。

**结论**：checkpoint 不可用（缺 node0 半份 + `.metadata`）。这是一个**结构性阻塞**——只要 node0 在存档时卡住，任何跑到终点的 100 步 run 都会死。已排除存储层（node1 本地盘 2 分钟证明很快）、已排除"单 rank 慢"（是 **node0 整节点 blocked**）。后续代码静态审查坐实崩溃点在 `megatron_checkpoint_manager.py:1077` 的 barrier（见 §6.4）。

---

## 六、关键发现与下一步

### 6.1 已确认的结论

1. **训练链路打通并通过验收**：SAPO + Megatron + vllm_ascend 在 16 × 910B3 上可稳定训练 100 步（task 716535 Succeeded），throughput 123.94 >100 ✓，reward +66% 向零 ✓，无 OOM/AssertionError/gloo timeout。
2. **throughput 破 100 的关键 lever = 三标志 dynamic_bsz**：probe5 micro8（静态最优）throughput 仅 80.9，probe6 追加 dynamic_bsz 后 113.2（+40%）。机制 = token 预算装箱（`rearrange_micro_batches`），最大增量在 `old_log_prob`（-82.5%），装箱后 rank 间计算量均衡。
3. **EP8 压通信有效**：AlltoAllV 绝对时长 -70%（16.1s→4.9s），total device time -66%（42.1s→14.4s）。
4. **offload 假设被证伪**：707489 关掉 verl 层 optimizer_offload，update_actor -1.5%（噪声级）；707032 三个 offload 全关则 OOM（HDO 是 Adam 态常驻 CPU 的唯一兜底，不可关）。
5. **RECOMPUTE=selective 不是解药**：probe3 仅 -10.4%（未达 >20% 阈值），且多用 7.6G 显存，生产仍用 full。
6. **checkpoint 落盘隐患仍在**：v0.8.0 上 sync barrier 崩溃 + async_save 坏掉（见 §6.4），验收 run 用 `SAVE_FREQ=-1` 规避，未产出可加载 ckpt。

### 6.2 探针完成情况（2026-08-11→13）

| 探针 | 目的 | 结果 |
|---|---|---|
| 692443 micro2 | micro_batch=2 验证 update_actor 下降 | update_actor 458s（-40%）|
| 703719 EP4/EDP4 | 测试 EP4 是否更优 | EP8 更优 |
| 707489 offload-off | 关 verl 层 optimizer_offload | -1.5%（噪声，offload 非主因）|
| 707032 full-offload-off | 三 offload 全关 | step1 OOM（HDO 不可关）|
| probe3 (710229) | RECOMPUTE selective | -10.4%（未达阈值）|
| probe4 (711571) | micro4 | -28.7%（行1命中）|
| probe5 (712691) | micro8 | -26.1%，throughput 80.9（行1命中，静态最优）|
| probe6 (712804) | + dynamic_bsz | -4.7%，throughput 113.2（行1命中，破 100 ✓✓）|

### 6.3 下一步规划

1. **recipe 产出**：基于 probe6 验收配置生成独立 recipe 脚本（归 verl-recipe 仓库 `sapo/` 目录），并在 `model_and_algorithm_support.md` 表中新增 SAPO 行（进行中）。
2. **checkpoint 修复**：在干净 v0.8.0 上复现 sync barrier 崩溃 + async_save 坏掉，按 §6.4 方案 1 加回 5 行 `async_calls_finalize_fn_exec`（待后续，PR 须人类提交）。
3. **可评测 checkpoint**：修复后跑 2 步 + `SAVE_FREQ=1` 冒烟（§6.4 验证清单），产出可加载 ckpt 以支持评测。

### 6.4 checkpoint 隐患专项定位（2026-08-13 代码审查，尚未实跑验证）

> **状态**：以下结论来自 v0.8.0 代码静态审查，**未经集群实跑冒烟验证**。用户指示：先记录问题与方案，后续再修；不确定是否为本地改动引入。任何修复前需先在干净 v0.8.0 上复现以排除本地 patch 干扰。

#### A. sync 路径崩溃点已坐实

`verl/utils/checkpoint/megatron_checkpoint_manager.py:1075-1077`：

```python
if not self.checkpoint_config.async_save:
    assert async_save_request is None, "..."
    torch.distributed.barrier()   # ← 682143 崩溃点
```

- 每次 `_save_dist_checkpoint` 写完一棵 dist_ckpt 树后紧跟一个 `barrier()`。
- `SAVE_CONTENTS=["model","extra"]` 排除 optimizer → **2 棵树 = 2 次 barrier**。
- 682143 的 30 分钟 gloo 超时即此 barrier：node0（rank0-7）76 分钟零分片，15 个 rank 在 barrier 上等满 1800s 后超时退出。
- **根因在进程内 IO 性质，非存储带宽**：裸写基准（task 681351，16 进程 × 4 GiB）JuiceFS 60-65 MiB/s/proc、聚合 ~1 GiB/s；但训练进程实测仅 1.64 MiB/s/rank（慢 38×），straggler 在裸写下不复现 → 慢和掉队都是训练进程内部的性质，存储/CPU/NUMA 已排除。

#### B. async_save 在 v0.8.0 是坏的（关键发现）

`async_save=True` 不仅不能救场，反而产出**不可加载**的 checkpoint：

1. **drain 方法被删**：`_dispatch_finalize`（`megatron_checkpoint_manager.py:1146-1167`）把 async 写请求塞进 Megatron `AsyncCallsQueue`，但负责排空队列的 `async_calls_finalize_fn_exec` 方法在 #6067（commit 044bbba2，workers→engines 迁移）时未从 `megatron_workers.py:984-988` 迁移到 `engine_workers.py`。原方法（5 行）：

   ```python
   @register(dispatch_mode=Dispatch.ONE_TO_ALL)
   def async_calls_finalize_fn_exec(self, blocking=False):
       from megatron.core.dist_checkpointing.strategies.base import async_calls
       async_calls.maybe_finalize_async_calls(blocking=blocking)
   ```

2. **trainer 调用点全 hasattr 守卫 → 静默 no-op**：`verl/trainer/ppo/ray_trainer.py:1430-1431, 1761-1762`：

   ```python
   if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
       self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
   ```

   方法不存在 → `hasattr` False → 整段跳过 → `AsyncCallsQueue` 从不排空。

3. **后果链**：队列不排空 → async 写请求堆积不完成 → 附在最后一个请求上的 `finalize_save_fn` 回调永不触发 → `_finalize_save`（`:1099-1144`）不执行 → `ckpt_contents.json` manifest、`latest_checkpointed_iteration.txt`、retention 全不写。`load_checkpoint`（`:856-987`）先读 `.metadata`，async 路径下 `.metadata` 也可能缺失 → **产物不可加载**。

4. **结论**：开 `async_save=True` 比 sync 更糟——sync 至少在 barrier 不超时时能写出完整产物；async 则结构性地产出残缺产物。**v0.8.0 上 async_save 不可用**。

#### C. 修复方案（待后续执行，本轮不修）

**方案 1（推荐，最小改动）**：在 `engine_workers.py` 的 `ActorRolloutRefWorker`（:665-668 `save_checkpoint` 附近）与 `TrainingWorker`（:426-428 `save_checkpoint` 附近）各加回上述 5 行 `async_calls_finalize_fn_exec`。`register`/`Dispatch` 在该文件已导入（:660 用了 `@register(dispatch_mode=Dispatch.ONE_TO_ALL)`），无需补 import。加回后 trainer 的 hasattr 守卫自动生效。

**方案 2（缓解，不改代码）**：100 步验收 run 用 `SAVE_FREQ=-1` 不落盘，彻底避开 barrier 面；checkpoint 保存作为独立短步冒烟单独验证。验收口径（throughput>100 + reward 上升）不要求末态 ckpt。

**验证方式（冒烟，待后续）**：2 步 + `SAVE_FREQ=1` + `SAVE_CONTENTS=["model","extra"]` + `MAX_ACTOR_CKPT_TO_KEEP=2`，跑后逐项检查：
1. 任务 Succeeded（非 barrier 超时）；
2. `global_step_N/actor/ckpt_contents.json` 存在（manifest，完整保存标志）；
3. `model/dist_ckpt/.metadata` + `__0_*.distcp`（非 0 字节）存在；
4. `extra/dist_ckpt/.metadata` + shard 存在；
5. rank0 vs rank1-15 shard 写入耗时（mtime 推算，确认 straggler 是否复现）；
6. load 回放：1 步 run + `load_checkpoint` 指向冒烟产物，确认可加载；
7. （async 专项）`latest_checkpointed_iteration.txt` 写入、日志无 "async request still pending"。

#### D. 不确定性声明

- 上述 A/B/C 均为 v0.8.0 静态审查结论，**未在集群实跑验证**。
- 本分支的 vllm patch 门控改动已在 rebase 到 main 时被上游 #7190/#7147 的无条件 patch 结构取代并删除；分支内现存改动（tau 路径、recipe、docs）均**未触碰 checkpoint 相关代码**，故 async_save 坏掉应是上游 v0.8.0 既存问题，非本地引入——但用户要求修复前先在干净 v0.8.0 复现确认。
- 修复（方案 1）属仓库侧代码改动，需 TDD 配套测试；项目 CLAUDE.md 禁止 agent 提 PR，须人类提交者逐行审阅 + 跑测试 + defend；提交前 `gh pr list --search "async_calls_finalize_fn_exec"` 查重。

---

*注：本报告基于任务 682143（早期失败 run）运行日志、节点文件实物核对、716535（验收 run）运行日志、探针实验（692443/703719/707489/707032/710229/711571/712691/712804）及 2026-08-13 代码静态审查整理。§6.4 为代码静态审查结论，未经集群实跑验证。*
