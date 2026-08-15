# SAPO × Megatron × NPU 训练速览

Last updated: 08/15/2026.

本文是 [SAPO × Megatron × NPU 运行报告](sapo_megatron_npu.md) 的精简版，只保留主要指标、训练配置和成功经验，供快速上手。完整数据、失败根因与探针细节见完整报告。

## 一、主要指标（task 716535，100 步验收 run，Succeeded）

| 指标 | 值 | 说明 |
|---|---|---|
| 任务状态 | **Succeeded**，约 12.8 小时，双节点 `exit code: 0` | 无 OOM / AssertionError / gloo timeout |
| `perf/throughput` | 均值 **123.94**，min 110.98 / max 147.16 | **50 读数全 >100 ✓**，前25 119.06 → 后25 128.83（+8.2% 缓升）|
| reward（`critic/rewards/mean`）| 前25 **-0.1474** → 后24 **-0.0496** | **+66% 向零**，收尾 step100 达全程最高 +0.148 |
| 显存峰值 | **29.22 / 60.96 GiB**（48%）| 跨全程恒定，无泄漏 |
| step 耗时 | ~450 s/步（无 profile 污染）| |

**两项验收判据均通过**：`perf/throughput > 100`（`metric_utils.py:669`，每卡每秒 token 数）+ reward 上升。

## 二、训练配置（probe6 验收基线）

集群：昇腾 910B3 × 16（2 节点 × 8 卡，HBM 60.96 GiB/卡）。模型 Qwen3-30B-A3B（128 experts），预转 mcore dist checkpoint。算法 SAPO（`policy_loss.loss_mode=sapo`），GRPO 采样，`use_kl_loss=False`。

| 类别 | 参数 | 值 | 说明 |
|---|---|---|---|
| 并行 | TP / PP / CP / EP / ETP | 4 / 1 / 1 / **8 / 1** | EP8 压通信：AlltoAllV 绝对时长 -70% |
| | GEN_TP | 4 | rollout 生成 TP |
| 训练批次 | TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE | 96 / 32 | 全局 batch / 每优化步 mini-batch |
| | PPO_MICRO_BATCH_SIZE_PER_GPU | **4** | 打包上限（dynamic_bsz 装箱）|
| 长度 | MAX_PROMPT_LENGTH / MAX_RESPONSE_LENGTH | 2048 / 4096 | |
| | PPO_MAX_TOKEN_LEN_PER_GPU | 8192 | token 预算（dynamic_bsz 装箱上限）|
| dynamic_bsz | actor.use_dynamic_bsz | **True** | throughput 破 100 的关键 lever |
| | rollout.log_prob_use_dynamic_bsz | **True** | 三标志必须同开（`engine_workers.py:561` 双向断言）|
| | ref.log_prob_use_dynamic_bsz | **True** | ref 静默不校验但必须一致 |
| SAPO | TAU_POS / TAU_NEG | 1.0 / 1.05 | 平滑温度（论文默认），正确路径 `actor_rollout_ref.actor.tau_pos` |
| | use_kl_loss / entropy_coeff | False / 0 | SAPO 无 KL |
| 优化器 | ACTOR_LR | 1e-6 | |
| | optimizer_cpu_offload + fraction | True + 1 | HybridDeviceOptimizer（HDO），不可关 |
| 内存 | megatron.param/grad offload | True | |
| | megatron.optimizer offload | False | 707489 证实 verl 层 optimizer_offload 冗余 |
| | RECOMPUTE | full | 最省显存（probe3 证实 selective 仅 -10.4% 不值得）|
| rollout | ROLLOUT_N / ROLLOUT_GPU_MEM_UTIL | 8 / 0.6 | |
| 存档 | SAVE_FREQ / MAX_ACTOR_CKPT_TO_KEEP | **-1** / 1 | 规避 sync barrier 崩溃面（见成功经验 5）|
| 步数 | total_training_steps / TOTAL_EPOCHS | 100 / 1 | |
| 调试 | PROFILE | **0** | 避免 3.3× profile 污染 |
| | EXPERIMENT_NAME | qwen3_30b_a3b_megatron_16npu_100step_probe6 | resume_mode=auto + 同名会捡旧 ckpt，必须换名 |

## 三、成功经验

1. **throughput 破 100 的关键 lever = 三标志 dynamic_bsz**。probe5 micro8（静态最优）throughput 仅 80.9，probe6 追加 dynamic_bsz 后 113.2（**+40%**），首次破验收线。机制 = `use_dynamic_bsz=True` 时 `rearrange_micro_batches`（`verl/workers/engine/utils.py:73-94`）按 token 预算装箱而非固定序列数切分，最大增量在 `old_log_prob`（173.9s → 30.5s，**-82.5%**），装箱后 rank 间实际计算 token 量均衡。三标志必须同开，否则启动崩或静默运行不同批处理方案。

2. **EP8 压通信有效**。TP4/EP8/ETP1 相对 TP4/EP4/ETP4：AlltoAllV 绝对时长 -70%（16.1s→4.9s），total device time -66%（42.1s→14.4s）。EP8 压缩的是 all-to-all 规模而非占比，通信仍是非计算瓶颈但绝对时长大降。

3. **offload 假设被证伪，HDO 不可关**。707489 关掉 verl 层 optimizer_offload，update_actor 仅 -1.5%（噪声级）→ verl 层 offload 冗余；707032 三个 offload 全关则 step1 OOM → HDO（Adam 态常驻 CPU）是唯一兜底，不可关。生产用 param/grad offload=True + optimizer offload=False + HDO=True。

4. **RECOMPUTE=selective 不是解药**。probe3 仅 -10.4%（未达 >20% 阈值），且多用 7.6G 显存。生产仍用 full（省显存，给 colocated rollout 留裕量）。

5. **checkpoint 落盘隐患用 `SAVE_FREQ=-1` 规避**。v0.8.0 上 sync 路径 `megatron_checkpoint_manager.py:1077` 的 `torch.distributed.barrier()` 会因进程内 IO 慢（1.64 MiB/s/rank vs 存储 60-65 MiB/s，慢 38×）拖爆 gloo 30 分钟超时；async_save 在 v0.8.0 也坏掉（drain 方法 `async_calls_finalize_fn_exec` 在 #6067 迁移时漏迁，trainer `hasattr` 守卫静默 no-op，产出不可加载）。验收 run 用 `SAVE_FREQ=-1` 不落盘彻底避开，验收口径（throughput>100 + reward 上升）不要求末态 ckpt。修复方案与冒烟清单见完整报告 §6.4。

6. **micro_batch 打包上限取 4**。probe4 micro2→4：update_actor -28.7%，显存零代价；probe5 micro4→8：-26.1% 但 throughput 80.9 未破 100；probe6 在 micro4 基础上追加 dynamic_bsz 才破 100。micro4 作打包上限 + dynamic_bsz 按 token 预算装箱是最终生产配置。

---

*精简版基于 task 716535（100 步验收 run）运行日志与探针实验（707489/707032/710229/711571/712691/712804）整理。完整数据与失败根因见 [SAPO × Megatron × NPU 运行报告](sapo_megatron_npu.md)。*
