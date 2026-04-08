
## 背景概述

随着大模型规模持续增长，推理与训练的性能瓶颈日益突出，尤其在MoE架构下，通信开销、算子效率与显存管理成为制约系统吞吐的关键因素。

本文基于Atlas 800T A2服务器，聚焦于Qwen3-30B-A3B模型在异步训练场景下的全链路性能优化，系统性地探索了从推理算子优化、FSDP训练加速到Fully-Async架构打通的完整技术路径。通过多维度调优，最终显著提升训练效率与资源利用率。


## 版本环境

- vLLM-Ascend: v0.11.0
- CANN: 8.3.RC1
- 模型：Qwen3-30B-A3B

## 推理性能优化：聚焦关键算子与通信瓶颈

### 专家并行切分策略优化（2倍以上收益）

在MoE模型推理中，专家并行（Expert Parallel, EP）与张量并行（Tensor Parallel, TP）的切分策略直接影响计算效率。早期版本中，EP分支因依赖复杂的`all2allv`通信与CPU-NPU同步，导致性能低于TP。经分析发现，当专家数量（ep）从2增至8时，EP2与EP8性能差距达2倍，而TP2与EP2在vLLM 0.11版本后趋于持平。

优化策略：根据专家数量动态切换切分方式——ep≤16时使用`all2all`，ep>16时启用`all2allv`。同时，通过调整并行策略，将`dp2ep2`切换为`tp2`，显著提升吞吐。

### Row Index 透传优化（2%收益）

`torch_npu.npu_moe_gating_top_k_softmax`算子输出的`row_idx`在后续`fused_experts`中可直接复用。原实现需额外执行`arange + transpose`生成索引，引入冗余计算。通过透传`row_idx`，避免重复计算，推理时间下降。

### 关闭GMM NZ 转换（1%收益）

尽管NZ（Non-Zero）格式理论上可减少显存读取开销，但在实际profiling中发现，`grouped matmul`算子并未启用NZ标志。注释掉相关转换代码后，推理性能提升1%。进一步分析发现，NZ转换主要影响权重加载阶段：在30B模型中，权重加载需执行128×3×48次ND-NZ转换。该开销在同步训练中显著影响吞吐，而在异步训练中可忽略。

> **注**：vLLM 0.11.0已通过环境变量`VLLM_ASCEND_ENABLE_NZ`支持动态控制，便于按场景灵活开关。

### HCCL OP EXPANSION MODE AIV（7%收益）

开启AIV（Asynchronous Intra-Node Vectorization）通信展开模式后，推理阶段通信时延显著降低。该模式通过优化HCCL通信调度，减少通信等待，实测推理性能提升约7%，为整体吞吐提升提供关键支撑。

---

## FSDP训练优化：从算子融合到资源调度

### Grouped Matmul 融合算子接入（训练时间减半）

初始profiling显示，训练中`free`时间高达61%，主要瓶颈集中在`nonzero`算子。该算子因输出shape不固定，需进行host-device同步，导致算子执行阻塞。

问题定位：MoE中通过`where`筛选token构建`selected_tokens × hidden_size`张量，无法避免同步。但`seq`在前向开始时已知，若能将所有乘法合并，即可避免shape不确定性。

解决方案：引入`Grouped Matmul (GMM)`融合算子，将同一专家的token聚类后统一计算。通过`argsort`与`index_select`实现token重排（permute），使显存切分在device侧完成。

> **参考接口**：[PyTorch Ascend API - GMM](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000880.html)
> **实现路径**：参考Megatron写法，使用`mindspeed`穿刺实现前向，后续版本已原生支持完整前反向。

优化后训练性能提升超过60%。

### Permute/Unpermute 融合算子（训练时间减少15%）

为消除`permute`与`unpermute`带来的额外通信与计算开销，引入融合算子：

```python
unpermuted_tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)
```

该融合算子将重排与反重排合并为单个NPU算子，减少中间张量生成与通信，训练时间进一步压缩至181s。

### Experts权重融合（训练时间减少15%）

模型原始权重按专家分散存储，导致`transpose`与`stack`算子耗时占比分别达27%与14%。通过预处理将所有专家权重提前`stack`合并，减少运行时计算。

> **实现方案**：参考VeOmni权重合并脚本，实现模型权重预处理。

同时，因vLLM权重加载器未适配新格式，需在`reshard`阶段重新拆分权重，确保兼容性。

### 切分策略优化（训练时间减少50%）

FSDP中通信量与batch数、权重规模正相关。通过将`sp size`从8降至2，显著减少通信总量，显著降低训练时间。

但纯FSDP存在负载不均问题：不同DP rank间显存占用差异大，易引发OOM。引入Ulysses负载均衡算法，使各rank分得相同head与序列，实现天然均衡。在保持token数不变前提下，提升`sp size`至8，训练时间明显降低。

### Jemalloc优化（训练时间减少7%）

引入jemalloc替代默认内存分配器，通过`LD_PRELOAD`注入`.so`库，有效降低host-bound开销。性能提升约7%。

---

## Megatron优化：显存控制与融合算子调优

### 调优成本降低

通过跳过`validate_non_overlapping_shards_metadata`校验与禁用`dist checkpoint`，显著减少权重加载时间。同时启用`rollout skip`，降低初始化开销。

### 融合算子与切分调优（训练时间减少50%）

Megatron在显存控制上具备优势。经验证，`tp4cp2ep8 + 激活函数重计算`为最优配置。同时，开启`rope`、`swiglu`、`permute-unpermute`等融合算子后，训练时间减少50%。

> **注**：部分融合算子在vLLM中未默认启用，需参考[MindSpeed](https://gitee.com/ascend/MindSpeed)实现。

### TND格式Host-Bound优化（预期提升30%）

TND格式下，`mindspeed`会将`rope`拆分为子序列，引发重复host操作，导致host-bound问题。预计在Megatron 0.15版本中，该问题将被修复，训练性能有望提升30%。

---

## Fully-Async 架构打通与性能调优

### Fully-Async 适配

原版Fully-Async使用`ray.utils.collective`进行reshard，但该接口暂不支持NPU。通过替换为`torch.distributed.broadcast`，并参考PR [verl#2924](https://github.com/volcengine/verl/pull/2924)完成适配，成功打通NPU支持路径。

### Partial Rollout + Staleness 优化（吞吐提升1倍）

在32卡配置下，开启`staleness=0.3`后，吞吐提升接近2倍收益。主要优势包括：

- **减少长尾响应**：`response max`降低一半，显著降低推理延迟。
- **填充推理空泡**：在空闲周期中注入更多样本，提升训练效率。

尽管`partial rollout`收益未显著显现，但`staleness`机制对整体吞吐提升贡献显著。

### Scaling 实验：推理扩展极限

为评估推理扩展能力，固定训练节点数为2，逐步增加推理节点数，结果表明，4节点为性能拐点，8节点提示推理扩展存在瓶颈。

### 虚拟内存启用（显存减少5G）

在Fully-Async架构中，rollout与update分离部署，无需频繁加载/卸载KVCache与权重。因此可安全启用虚拟内存：

```bash
PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
```

在64卡任务（32卡rollout + 32卡update）中，显存从45G降至40G，显著提升资源利用率。

### 资源配比优化（性能提升10%）

在Fully-Async场景下，`timing_s/update`与`timing_s/gen`存在跷跷板效应。通过Scaling实验发现，推理性能在32→48卡间可线性提升。因此将训推资源比从1:1调整为3:1，吞吐量提升约10%。

---

## 训练后端选型决策

尽管Megatron在显存控制与算子融合方面表现优异，但其需依赖`rollout importance sampling`补偿训推差异，增加系统复杂性。在Fully-Async场景下，FSDP方案更易实现与框架对齐，最终选择FSDP作为主训练后端。

---

## 总结与展望

本实践系统性地完成了从推理算子优化、FSDP训练加速到Fully-Async架构打通的全链路调优。核心经验包括：

- **算子融合**是突破性能瓶颈的关键，尤其在MoE与GMM场景下。
- **通信模式优化**（如AIV、GMM）对推理与训练均有显著收益。
- **架构设计**需权衡显存、通信与负载均衡，FSDP在异步场景更具优势。
- **虚拟内存与资源配比**是提升大规模训练效率的重要手段。

未来将持续探索Eagle3、CP特性与EPD分离等新技术，进一步释放硬件潜力。


