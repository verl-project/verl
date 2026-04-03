
## 一、背景概述

随着大模型后训练范式从SFT向SFT-RL-SFT演进，强化学习在大模型对齐与能力提升中扮演关键角色。基于昇腾NPU平台的Verl框架已成为主流训练工具之一，尤其在长序列推理场景下对性能与显存效率提出更高要求。

本文基于Atlas 800T A2服务器，聚焦于Qwen3-235B规模模型在2k输入、30k输出长度下的强化学习训练性能优化，针对推理阶段耗时过长、显存压力大、训练中断风险高等问题，系统性地开展性能分析与调优。

### 版本信息

性能优化版本（部分非版本号为master分支commit id）：

| 组件        | 版本            |
| ----------- | --------------- |
| HDK         | 25.2.1          |
| CANN        | 8.3.RC1         |
| VeRL        | 21271aa         |
| vllm        | release/v0.11.0 |
| vllm-ascend | 15b2e5c         |
| MindSpeed   | 35da6ac         |
| Megatron-LM | core_v0.12.1    |
| torch       | 2.7.1           |
| torch-npu    | 2.7.1-0919      |

MindSpeed-RL 2.2.0商发配套版本：

**链接：**

| 组件        | 版本                                     |
| ----------- | ---------------------------------------- |
| HDK         | 25.2.1                                   |
| CANN        | 8.3.RC1                                  |
| VeRL        | 796871d7d092f7cbc6a64e7f4a3796f7a2217f5e |
| vllm        | 38217877aa70041c0115ee367b75197af9cbc5ad |
| vllm-ascend | 1de16ead8eecfec8903ec1b330b27a4fa2593c35 |
| MindSpeed   | 1cdd0ab                                  |
| Megatron-LM | core_v0.12.1                             |
| torch       | 2.7.1                                    |
| torch-npu    | 2.7.1                                    |

## 二、性能瓶颈分析

在初始配置下，经profiling拆解发现，**generate阶段耗时较大**，成为核心瓶颈。主要问题集中在以下三方面：

1. **输入配置不合理**：batch_size × n_samples = 384 × 16 = 6144，远超卡数整除能力，导致数据分发效率低下；
2. **推理性能未充分优化**：部分关键特性未启用，如aclgraph图模式、融合算子等；
3. **训练阶段配置缺失**：update阶段占比过高，缺乏分布式优化器、ETP、TND子序列Batch Rope等关键优化手段。

## 三、性能优化方案

### 3.1 通用性能优化

#### 3.1.1 二级流水（Task Queue Level 2）

通过设置环境变量 `export TASK_QUEUE_ENABLE=2`，启用二级流水算子下发机制。该优化将算子任务拆分为一、二级流水并行执行，尤其将workspace相关任务迁移至二级流水，显著掩盖Host调度延迟，提升整体端到端性能。**该配置仅在二进制场景生效，建议优先使用Level 2**。

#### 3.1.2 高性能内存管理（jemalloc）

在ARM架构环境下，启用jemalloc可有效降低内存碎片，提升内存分配效率。安装方式如下：

```bash
apt install libjemalloc2
export LD_PRELOAD="$LD_PRELOAD:/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
```

> 注：x86平台无明显收益，仅在ARM环境推荐使用。

---

### 3.2 推理性能优化

#### 3.2.1 ACLGraph图模式（PIECEWISE）

VLLM后端支持eager与图模式切换。通过将 `enforce_eager=False` 启用ACLGraph，可将多个小算子融合为图执行，减少Host下发开销。实测端到端性能提升达15%-20%。

> **关键修复**：早期版本因MindSpeed patch导致图捕获异常，已通过上下文管理器（contextmanager）机制修复，确保在rollout阶段动态切换 `torch.compile` 与 `dummy_compile`，避免通信异常。

#### 3.2.2 大专家并行（EP）配置优化

开启 `enable_expert_parallel=True` 可提升专家层并行度，但需注意：

- 依赖CANN 8.3.RC1及以上版本；
- A2平台通信带宽有限，EP过大（如128）反而导致通信开销激增，性能劣化；
- 建议根据实际通信能力合理设置EP大小，避免“过切”。

---

### 3.3 训练性能优化

#### 3.3.1 融合算子优化

通过启用多个融合算子，减少算子下发次数，提升计算效率：

| 算子类型 | 配置项 | 说明 |
|--------|--------|------|
| RMSNorm | `use_fused_rmsnorm=True` | 替换原始分步归一化 |
| SwiGLU | `use_fused_swiglu=True` | 融合GELU与线性变换 |
| RoPE | `apply_rope_fusion=True` + `use_fused_rotary_pos_emb=True` | 将7个算子融合为1个，耗时从86μs降至24μs，端到端收益约0.5% |
| GMM | `moe_grouped_gemm=True` | 融合多专家计算，提升GMM算子效率 |

#### 3.3.2 分布式优化器

启用分布式优化器可显著降低显存占用与通信压力：

```yaml
actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
actor_rollout_ref.ref.megatron.use_distributed_optimizer=True
actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
```

#### 3.3.3 ETP（Expert Tensor Parallel）优化

设置 `expert_tensor_parallel_size=1`，避免专家参数在行列维度切分，提升小专家场景下的计算效率，尤其适用于MoE模型。

#### 3.3.4 TND场景Batch Rope优化

在TND（Token-Nested-Data）格式下，原始方案对每条子序列独立执行RoPE，存在循环开销。新方案通过生成统一频率矩阵，实现一次RoPE完成全部编码，**RoPE阶段耗时减少50%**，端到端性能提升1%-2%，且序列越长收益越显著。

#### 3.3.5 训练并行策略调优

原始配置中CP（Context Parallel）过大，PP通信占比高达56.6%，成为瓶颈。通过以下调整优化：

| 优化项 | 原配置 | 新配置 | 效果 |
|------|------|------|------|
| CP | 8 | 4 | 降低上下文通信 |
| PP | 12 | 8 | 显著减少PP通信 |
| EP | 8 | 16 | 缓解显存压力 |
| ETP | 未开启 | 1 | 提升专家计算效率 |

优化后PP通信占比从56.6%降至29.55%，整体通信效率显著提升。

---

## 四、关键问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **长序列推理超时** | HCCL通信超时 | 将输入与输出batch的`interleave`均设为`False`，避免长尾问题 |
| **精度异常（reward跌0）** | 仅修改输入batch interleave，未同步输出batch | 保持输入与输出batch排布一致 |
| **开启CP报错** | 缺少`override_transformer_config.context_parallel_size`配置 | 补全配置项 |
| **group_type报错** | torch-npu与mindspeed版本不匹配 | 升级至兼容版本 |
| **找不到index_first_axis** | rl-plugin未正确patch | 手动修改`verl/__init__.py`文件 |
| **训练拉起慢** | 校验函数阻塞 | 临时返回空值，加速启动 |
| **DataProto报错** | batch_size × n_samples 不能被卡数整除 | 调整为可整除组合（如384×16） |
| **tensordict精度问题** | `.to(cpu)`为非阻塞操作 | 升级tensordict至0.10.0及以上版本 |
| **开启PP报错** | `megatron.training`模块缺失 | 升级mindspeed至9.5后版本，或修复`optimize_p2p_comm.py` |
| **log_prob阶段OOM** | 30k序列下显存压力过大 | 降低`log_prob_max_token_len_per_gpu`至16k，平衡显存与性能 |

> **建议**：`log_prob_max_token_len`不宜过小，否则输入shape过细，计算效率下降。

---

## 五、新模型适配与复用

目标模型与Qwen3-MoE结构高度一致，仅在Attention部分省略两个Norm层，其余结构、权重格式与配置基本兼容。适配过程参考已有Qwen3-MoE方案，**所有优化配置可直接迁移**。

适配后性能与原模型基本一致，验证了优化方案的通用性与可复用性。

---

## 六、总结与展望

本项目通过系统性分析与多维度调优，核心优化路径包括：

- **推理侧**：启用ACLGraph图模式、融合算子、Batch Rope；
- **训练侧**：启用分布式优化器、ETP、合理调整并行策略；
- **配置侧**：优化batch排布、显存控制、版本兼容性。

未来可进一步探索**训推异步方案**（如VeRL Fully Async）、**更高效的图模式**（如Torch-AIR）以及**动态切分策略**，持续提升大规模模型训练的效率与稳定性。



