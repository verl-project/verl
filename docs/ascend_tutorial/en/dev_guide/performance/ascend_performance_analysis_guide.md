# Ascend Performance Analysis Guide

Last updated: 02/24/2026.

## Background

Following the release of DeepSeek-R1, large-model reinforcement learning (RL) training has attracted widespread attention. In the Ascend NPU environment, the verl framework has accumulated rich experience in performance tuning. This document systematically summarizes the methodology for performance data collection and analysis, aiming to help developers use the MindStudio toolchain more efficiently and achieve performance optimization in reinforcement learning scenarios.

### Overview of the Reinforcement Learning Computation Flow

1. **Rollout**: The policy (actor) model performs inference based on the input prompt sequence to generate a response (response sequence).
2. **ref logprob**: Based on the prompt and the generated response, the reference model computes the ref logprob used for KL divergence computation.
3. **logprob**: Based on the prompt and the generated response, the actor model computes the logprob used for importance sampling.
4. **reward**: Based on the prompt and the generated response, the reward model evaluates the reward value R_N.
5. **update**: Based on the computed R_N, ref logprob, and logprob, compute the optimization function and policy gradient, and update the actor model.

![rl_data_stream](https://github.com/wucong25/verl-data/blob/main/en/figures/rl_data_stream.png)

## Enabling the Profiling Tool

### How to Enable

For the enabling and configuration tutorial, refer to: [Profiling Collection Guide](./ascend_profiling.rst)

## Performance Analysis Methodology

### Overall Performance Overview Analysis

#### 1. Long-Duration Task and Resource Idle Analysis

- **Operation**: Use MindStudio Insight to load the profiling data, automatically identify different computation stages, and use the RL tab pipeline diagram to locate long-duration tasks and NPU resource idles (bubbles).
- **Value**: Quickly grasp the time proportion of different stages.
- **Demo**:

![Bubble_analysis](https://github.com/wucong25/verl-data/blob/main/en/figures/Bubble_analysis.png)

#### 2. Load Balancing Analysis

- **Operation**: Directly view the MSTX annotation data through MindStudio Insight to observe the load balancing across different DP Ranks in the Rollout phase.
- **Value**: Quickly identify load imbalance issues.
- **Demo:**

![Load_Balancing_Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Load_Balancing_Analysis.gif)

#### 3. Cluster Overall Performance Analysis

- **Operation**: Combine the rl_analysis feature of MSTT to generate a cluster Timeline thumbnail and observe the overall time of each stage.
- **Value**: Macroscopically grasp cluster performance bottlenecks.
- **Guide**: [rl_analysis documentation](https://gitcode.com/Ascend/mstt/blob/pre-research/profiler/msprof_analyze/docs/features/rl_analysis.md)
- **Demo**:

![Cluster%20Performance%20Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Cluster%20Performance%20Analysis.png)

### Fine-Grained Analysis

#### Performance Analysis

- **Operation**: Profiling data can be loaded using MindStudio Insight Windows or Linux version.
- **Value**: MindStudio Insight supports analyzing task scheduling efficiency, operator execution performance, compute resource utilization, collective communication performance, and so on Its Timeline view provides task decomposition and Overlap analysis capabilities (**a unique core feature of MindStudio, unavailable in NVIDIA and other competitors, and an essential tool for AI tuning**), and supports mouse-interactive analysis.
- **Demo**:

![performance%20analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/performance%20analysis.png)

#### Memory Analysis

##### **Analyzing System Memory Changes with Profiling Combined with the Call Stack**

- **Operation**: Enable the call stack and memory view features when collecting data.
- **Value**: Observe framework and CANN memory allocation/release, and trace back to the front-end Python code through the call stack.
- **Demo**: Analyze memory changes in combination with the call stack. The effect is shown below:

![in-memory%20analytics](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/in-memory%20analytics.gif)

##### **Deep Memory Analysis with the msleaks Tool**

- **Steps**: Refer to the [msleaks tool usage guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/msMemScope/docs/en/user_guide/memory_analysis.md).
- **Value**: You can view the framework memory allocation total line chart/memory block diagram, and directly map it to the call stack, enabling deep analysis of framework memory usage.
- **Demo**:

![msleaks](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/msleaks.gif)

## Performance Analysis Cases

For detailed performance analysis, profiling must enable **level1**; otherwise, key operator information will be missing.

### 1. Host Bound Diagnosis

Host bound refers to the phenomenon where the total CPU task volume exceeds that of the NPU, resulting in idles (bubbles) during NPU execution. You can determine this by looking at the Host2Device synchronization lines: if the lines are all skewed, it proves that the set signal comes earlier than the wait signal, and the NPU executes as soon as it is ready, which is actually device bound:

![host_bound_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_1.png)

If diagnosed as host bound, we can open the CPU side and find the dispatch time of each operator. Note that you need to find the cumulative value of all CPU time, not a single layer, because the first call takes a long time. For example, for GmmSwigluQuant in the figure below, the first call on the CPU takes 1ms, and each subsequent call only takes 200μs.

![host_bound_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_2.png)

At this point, some operators are carrying a heavy load while others are dragging behind, and the latter outnumber the former. We prioritize **finding the top operators whose host time exceeds the device time; these are the operators dragging things behind**, and they can be handed over to the operator team for focused analysis.

### 2. Network Architecture Rationality Analysis

Sometimes the model network is not organized in the most efficient way, and this is very easy to identify in profiling. The analysis approach is introduced below with examples.

Generally speaking, the large hot operators in an LLM are the matrix multiplication computations in Attention and FFN. Together, they can account for 70%+ of the computation time in prefill and 50%+ in decode. If the overall time proportion does not meet expectations, or new faces appear in the profiling, or there are too many concatenation-type operators, it is worth analyzing the model network to see whether the operators are being used incorrectly. Concatenation-type operators in particular are worth analyzing one by one.

For concatenation-type operators like slice/split/concat and conversion operators like transpose/cast, their presence is often caused by the preceding and following operators not being directly compatible. If the preceding operator can directly perform tail-processing on the output, it can often save the startup overhead of one operator and one redundant read/write. However, such a change may not conform to the basic design principles of operators.

As a positive example, suppose the output shape of a Matmul is [m, n0 + n1], and we then attach two slices, both taking this [m, n0 + n1] tensor as input and producing outputs of [m, n0] and [m, n1] respectively. The first optimization idea is to change the two slices into a single split, which roughly halves the time and allows the [m, n0 + n1] memory to be released earlier. A further optimization idea is to split the matrix multiplication weight from [k, n0 + n1] into [k, n0] and [k, n1], splitting the original matrix multiplication task into two (provided the combined time of the two is not much worse than before, and the core-splitting strategy does not cause issues), thereby completely eliminating the slice/split operation.

![network_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_1.png)

As a negative example, consider Rmsnorm(fp16)+Cast(fp16->fp32)+Matmul(fp32). Although Rmsnorm has fp16 inputs and outputs, internally it computes in fp32 to preserve the precision of accumulation. If the Cast is fused into Rmsnorm, the Rmsnorm—which already uses fp32 internally—can save the trailing fp32->fp16 cast, and combined with the eliminated Cast, we save two casts and avoid one precision loss. Although this seems to improve both precision and performance, an Rmsnorm with fp16 input and fp32 output violates the principle (the core input and output should be of the same data type). Unless we can frequently find such a structure in widely-used open-source models, proving its universality, the operator team will not allow such an operator.

![network_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_2.png)

### 3. Operator Performance Initial Diagnosis

You need to use `"./ASCEND_PROFILER_OUTPUT/operator_details.csv"` for analysis to determine whether an operator has a performance issue.

The Profiling tool computes the average busy time of these pipelines on different cores (xxx_time), divides it by the full kernel time of the slowest core (task_duration), and obtains the pipeline utilization (xxx_ratio). Although these pipelines depend on each other and the transfer-type pipelines compete for bandwidth, as long as the operator is properly designed, they can be mutually masked. Therefore, we can preliminarily consider that **when an operator's execution time is large enough, it should form a bound on one pipeline**, that is, the utilization should be high enough. Empirically, when a single operator takes 50μs, it can be considered that the operator should be on the bound pipeline, achieving 80%+ occupancy.

Taking the figure below as an example, the first row is an FA operator and the second row is a Matmul operator. FA achieves 88.1% utilization on the vec pipeline, and the Matmul operator achieves 89.8% utilization on the mac pipeline. Their performance can be considered acceptable.

![Operator%20performance](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Operator%20performance.png)

### 4. Affinity Shape Adjustment

For a model, we cannot control the hyperparameters, but we can control factors such as concurrency, weight format, and splitting strategy to accommodate the operator and let it perform at its best. This section focuses on two aspects—operator transfer efficiency and load balancing—and discusses directions worth trying on the model side.

#### 4.1 Transfer-Efficiency-Affinity Shape

mte2 is a pipeline whose own efficiency is severely affected by shape. To ensure maximum transfer efficiency for mte2, at least one of the following two conditions must be met:

**(1) The transferred matrix uses nz as format (optimal)
(2) The tail axis of the transferred matrix is 512B-aligned and not an integer multiple of 16KB (approximately optimal)**

For weight matrices, in the inference phase—especially decode—we usually satisfy (1), and in the training phase we usually satisfy (2). **If we cannot achieve (1), we should accommodate (2)**. Typical approaches include:

1. If (1) is not achieved and the matrix's leading axis is affinity-friendly while the tail axis is not, transpose it.
2. Adjust the TP splitting strategy to avoid an unfriendly tail axis.

#### 4.2 Load-Balancing-Affinity Shape

When the operator shape is small, due to operator semantics, we may not be able to utilize all cores, or even with all cores enabled, load balancing is poor. This subsection mainly analyzes the small shapes in the decode phase.

First, we clarify how many cores the current NPU card has. If you are not sure, the numbers like 20 and 40 in the profiling output indicate 20 cores; otherwise, it is 24 cores. Here, 24 cores actually represents a group consisting of one cube and two vectors. We can consider it as one cube acting as the master core, with two vectors as slave cores. If an operator is a pure vector operator, the concept of groups no longer applies; the 40 or 48 vector cores act as master cores and directly take logical tasks independently.

For vector operators in LLMs, one common core-splitting strategy is to split along the highest dimension, that is, the batch dimension, which is common in norm-type and dynamic-quantization-type operators that perform reduction on low dimensions (also called tail axis). The other is to flatten everything, allowing operators to be split very finely, such as elementwise operators. For the first type, we can focus on load balancing issues on the model side. For example, if we set a batch of 48 but the hardware has 40 vector cores, these 40 cores will loop twice, and in the second loop most cores will have nothing to do. This batch size can be considered unfriendly. If the batch is set to 64 or 80, performance is expected to be lossless. In the same situation, if it is a 48-core card, we can consider this a very friendly batch size.

For cube-type operators, the common core-splitting strategy is to split M and N by base (the K axis is the accumulation axis, and splitting on it introduces determinism issues). The most common blocking is baseM=128, baseN=256. In the decode phase, the time can basically be considered as all spent transferring weights, because the activation M is extremely small, and the M direction most likely has only one block, so the right matrix only needs to be transferred once. Therefore, within the range of M≤128, we can freely increase M with basically no performance loss. If M exceeds 128, (128, 256] can be considered the next performance tier.
Besides M, the tasks split on the N axis also affect operator affinity. Taking the MLA preprocessing in deepseekR1 as an example, it uses the same activation (shape [batch_size, 7168]) to do matrix multiplication with two weights (shapes [7168, 1536] and [7168, 576]). When batch_size cannot be made large, even if baseN is reduced to 128, the N axis still cannot use up all cores. So the time of these two matrix multiplications each is approximately equal to the time of multiplying the combined N axis of the weights (shape [7168, 2112]). If we only consider model competitiveness, we prefer to merge these two weights; otherwise, the bandwidth utilization of both small matrix multiplications will be very poor.

For Attention operators, the common core-splitting strategy is q_seqlen, batch_size, and kv_headnum. In the incremental phase, q_seqlen is merged by MTP and GQA multiples, but usually still does not exceed 128, so a second task cannot be split out, and the parallelism is basically batch_size * kv_headnum.

Overall, based on shape information and operator category, we can identify whether an operator has a load balancing issue, and therefore anticipate our splitting-strategy choices and the batch strategy for maximum throughput.
