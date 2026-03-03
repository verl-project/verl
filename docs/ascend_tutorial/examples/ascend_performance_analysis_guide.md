# Ascend Performance Analysis Guide

Last updated: 02/24/2026.

## 背景介绍

随着DeepSeek-R1的发布，大模型强化学习（RL）训练受到广泛关注。在昇腾NPU环境下，verl框架已积累了丰富的性能调优经验。本文系统总结了包括性能数据采集与分析在内的方法论，旨在帮助开发者更高效地运用MindStudio工具链，实现强化学习场景下的性能优化。

### 强化学习计算流程概述

1. **Rollout**：策略（actor）模型基于输入的prompt序列，推理生成回答（response序列）
2. **ref logprob**：基于prompt和生成的response，reference模型计算ref logprob用于KL散度计算
3. **logprob**：基于prompt和生成的response，actor模型计算logprob用于重要性采样
4. **reward**：基于prompt和生成的response，奖励模型评估奖励值R_N。
5. **update**：基于计算得到的R_N、ref logprob、logprob计算优化函数和策略梯度，对actor模型进行更新

![rl_data_stream](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/rl_data_stream.png)

## profilling工具使能

### 使能方法

使能和配置教程可参考：[verl/docs/ascend_tutorial/ascend_profiling_zh.rst at main · verl-project/verl](https://github.com/verl-project/verl/raw/main/docs/ascend_tutorial/ascend_profiling_zh.rst)

## 性能分析方法论

### 整体性能概览分析

#### 1. 长耗时任务与资源空泡分析

- **操作**：使用MindStudio Insight加载profiling数据，自动识别不同计算阶段，通过RL页签流水图定位长耗时任务与NPU资源空泡
- **价值**：快速掌握不同阶段耗时占比
- **效果展示**：

![Bubble_analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Bubble_analysis.png)

#### 2. 负载均衡分析

- **操作**：通过MindStudio Insight直接查看MSTX打点数据，观察Rollout阶段不同DP Rank的负载均衡情况
- **价值**：快速识别负载不均问题

- **效果展示：**

![Load_Balancing_Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Load_Balancing_Analysis.gif)

#### 3. 集群整体性能分析

- **操作**：结合MSTT的rl_analysis功能，生成集群Timeline缩略图，观察各阶段整体耗时
- **价值**：宏观掌握集群性能瓶颈
- **操作指南**：[rl_analysis使用文档](https://gitcode.com/Ascend/mstt/raw/pre-research/profiler/msprof_analyze/docs/features/rl_analysis.md)
- **效果展示**：

![Cluster%20Performance%20Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Cluster%20Performance%20Analysis.png)

### 细粒度分析

#### 性能分析

- **操作**：可通过 MindStudio Insight Windows 或 Linux 版本加载 Profiling 数据

- **价值**：MindStudio Insight 支持分析任务调度效率、算子执行性能、计算资源利用率、集合通信性能等。其 Timeline 视图具备任务拆解与 Overlap 分析功能（**为 MindStudio 独有核心特性，在 NV 及其他竞品中不具备，是 AI 调优的必备工具**），并支持鼠标交互式分析。

- **效果展示**：

![performance%20analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/performance%20analysis.png)

#### 内存分析

##### **通过 Profiling 结合调用栈分析系统内存变化**

- **操作**：采集数据时开启调用栈和内存视图功能。
- **价值**：观察框架、CANN内存申请释放情况，可结合调用栈跟踪到前端python代码。
- **效果展示**：结合调用栈进行内存变化分析。效果如下所示：

![in-memory%20analytics](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/in-memory%20analytics.gif)

##### **使用 msleaks 工具进行深层次内存分析**

- **操作步骤**：参考 [msleaks 工具使用指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/msleaks/atlas_msleaks_0001.html)。
- **价值**：可以查看框架内存申请总量折线图/内存块图，并直接对应调用栈，可深层次分析框架内存使用情况。
- **效果展示**：

![msleaks](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/msleaks.gif)

## 性能分析案例

要做具体的性能分析，profiling要开启**level1**，否则算子的关键信息会缺失。

### 1.host bound诊断

host bound是指CPU任务量综合大于NPU，导致NPU执行出现空泡的现象。可以通过看Host2Device的同步连线来判断，如果连线都是歪的，那证明这里的set信号早于wait信号，NPU一ready就执行了，那也是device bound：

![host_bound_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_1.png)

如果确诊为host bound，那么我们可以打开CPU侧，找出各算子的下发耗时。注意找的时候需要找出所有CPU耗时的累加值，而不能找单层，因为首次调用的耗时是很长的。例如下图的GmmSwigluQuant，CPU上首次调用需要1ms，后续每次只需要200us。

![host_bound_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_2.png)

此时有的算子在负重前行，有的算子拖了后腿，后者多于了前者。我们优先**找出来host耗时大于device的top算子，这些算子是拖后腿的**，可以交予算子团队重点分析。

### 2.组网合理性分析

有的时候，模型组网没有按照最高效的方式来，这一点在profiling中是非常易于识别的，下面会介绍一下分析思路并给出例子。

通常来讲，LLM中的大的热点算子是Attention和FFN中的矩阵乘计算，二者加起来在prefill下可能达到计算耗时的70%+，decode下可能达到50%+。如果整体的耗时比例不符合预期，或者profiling中出现了一些新面孔，或者拼接类算子太多了，这都值得我们去分析一下模型组网，是不是使用算子的方式错了？尤其是拼接类算子，是值得我们逐一分析的。

对于slice/split/concat这样的拼接类算子，还有transpose/cast这种转换算子，他们的存在往往是前后算子不直接配套造成的。如果前一个算子可以直接对输出做好尾处理，往往可以节省一个算子的启动开销和一次冗余读写。但这样的改变不一定符合算子的基本设计原则。

举一个正例，对于某次Matmul的输出shape为[m, n0 + n1]，在这后面我们接了两个slice，输入均为这个[m, n0 + n1]的tensor，输出分别为[m, n0]和[m, n1]。第一个优化的思路是将两个slice改为一个split，这样耗时可以基本减半，[m, n0 + n1]的显存也可以尽早释放。进一步优化的思路是将矩阵乘的权重从[k, n0 + n1]分割为[k, n0]和[k, n1]，将原来的矩阵乘任务分成两个（前提是这两个的耗时加起来不比之前的劣化太多，分核策略不能出问题），从而彻底消除这个slice/split操作。

![network_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_1.png)

举一个反例，Rmsnorm(fp16)+Cast(fp16->fp32)+Matmul(fp32)，Rmsnorm虽然输入输出都是fp16，但考虑到累加运算的精度，内部是fp32做计算的。如果将Cast融到Rmsnorm内，本就内部使用fp32做计算的Rmsnorm就可以省去一个末尾fp32->fp16的cast，加上我们干掉的Cast，总共节省两个cast的同时避免了一次精度丢失。虽然这样看起来精度性能双收了，但fp16进，fp32出的Rmsnorm是反原则的（核心输入和输出需要是同数据类型），除非我们能在广大开源模型中频繁找到这样的结构，证明它的普适性，否则算子团队是不允许做这样的算子的。

![network_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_2.png)

### 3.算子性能初诊

需要利用`".\ASCEND_PROFILER_OUTPUT\operator_details.csv"`来做分析，从而判断算子识否有性能问题。

Profiling工具会统计这些流水线在不同核上的平均繁忙时间（xxx_time），与最慢核的完整kernel耗时（task_duration）做除法，得到流水线利用率（xxx_ratio）。这些流水线之间虽然互有依赖，且搬运类流水线会互抢带宽，但算子只要设计得当，是可以做到互相掩盖的。因此我们可以初步认为，**当算子的执行耗时大到一定程度上，算子应当在某一条流水线上形成bound**，即利用率要高到一定程度。经验上，在单算子耗时达到50μ时，就可以认为算子应当在bound流水线上，达成80%+的占用率了。

以下图为例，第一行是一个FA算子，第二行是一个Matmul算子，FA在vec流水线上达到了88.1%的利用率，Matmul算子在mac流水线上达到了89.8%的利用率，他们的性能可以认为是合格的。

![Operator%20performance](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Operator%20performance.png)



### 4.亲和shape调整

对于一个模型而言，超参是我们控制不了的，但我们可以控制并发度、权重格式、切分策略等因素来迎合算子，使其发挥出最大的性能，这一节主要从算子搬运效率和负载均衡两个方面出发，讨论模型侧值得尝试的调整方向。

#### 4.1 搬运效率亲和的shape

mte2是一个自身效率严重受shape影响的流水线。要想让mte2保证最大搬运效率，我们需要保障如下两个条件至少满足其一：

**（1）被搬运的矩阵使用nz作为format（最优）
（2）被搬运的矩阵的尾轴512B对齐，且不为16KB的整数倍（近似最优）**

对于权重矩阵来说，推理阶段尤其是decode，我们通常满足（1），训练阶段我们通常满足（2）。**如果我们做不到（1），我们就要迎合（2）**。典型的手段有：

1，如果没达成B的矩阵的首轴是亲和的而尾轴不亲和，那么对它做transpose
2，调整TP切分策略，避免出现不亲和的尾轴

#### 4.2 负载均衡亲和的shape

在算子shape不大时，受制于算子语义，我们有可能不能把所有核都利用起来，或者即使开满核，负载均衡却很差。这一小节主要是对decode阶段的小shape做分析。

首先，我们明确出当前NPU卡是多少核的，如果不清楚，跑出来的profiling里都是20，40这样的数，就说明是20核，反之是24核。这里我的24核其实是代表了一个cube和两个vector组成的小组，我们可以认为是一个cube作为主核，带了两个vector作为从核。如果一个算子是纯vector算子，那么就不再有组的概念，40或48个vector核会作为主核直接独立去拿逻辑任务。

对于LLM中的vector算子，它的一种常见分核策略有可能是分在最高维，也就是batch维，常见于对低维（也叫尾轴）有规约操作的norm类、动态量化类等算子；另一种是整体拍平，允许算子切分的非常细的算子，如elementwse算子。对于第一种，我们就可以在模型侧关注它的负载均衡问题。例如我们打48batch，而硬件却是个40个vector核，那这40个核会循环2次，第二次有多数的核会无事可做，这个batch数就可以认为是不友好的。如果将batch打到64或80，性能可以预见会是无损的。同样的情况下，如果是48核的卡，那我们可以认为这就是个非常友好的batch数。

对于cube类算子，它常见的分核策略是以base快去切分M和N（K轴是累加轴，对它分核会引入确定性问题）。最常见的分块是baseM=128，baseN=256。在decode阶段，我们的耗时基本可以看做都是在搬权重，这是因为激活的M极小，M方向大概率只分了一块，那么右矩阵就只需要搬一次。所以我们在M≤128的范围内可以尽情提高M，对性能都基本是无损的，如果M大于128，可以认为(128, 256]是下一个性能分档。
除了M外，N轴切分的任务也影响算子亲和性，以deepseekR1中的MLA预处理为例，它会使用同一个激活（shape为[batch_size, 7168]）与两个权重做矩阵乘(shape为[7168, 1536]和[7168, 576])。在batch_size打不大的情况下，即使baseN缩短为128，N轴都不能用满核数，所以此时这两个矩阵乘各自的耗时，会约等于将他们权重N轴拼起来乘(shape为[7168, 2112])的矩阵乘的耗时。如果仅考虑模型竞争力，我们更希望对这两个权重做合并，否则两个小的矩阵乘带宽利用率都会非常差。

对于Attention算子，它常见的分核策略是q_seqlen、batch_size和kv_headnum。增量阶段q_seqlen会以MTP和GQA倍数做合并，但是通常也不会大过128，划分不出第二个任务，那么并行度基本就是batch_size * kv_headnum。

总的来说，我们可以依据shape信息和算子类别，对算子是否有负载均衡问题作出识别，从而对我们切分策略选择，最高吞吐量的batch策略作出预判。











