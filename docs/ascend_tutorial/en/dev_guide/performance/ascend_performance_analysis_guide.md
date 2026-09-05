# Ascend Performance Analysis Guide

Last updated: 02/24/2026.

## Background

With the release of DeepSeek-R1, reinforcement learning (RL) training for large language models has attracted widespread attention. In the Ascend NPU environment, the verl framework has accumulated extensive experience in performance tuning. This guide systematically summarizes a methodology that includes performance data collection and analysis. It aims to help developers use the MindStudio toolchain more efficiently to achieve performance optimization in reinforcement learning scenarios.

### Reinforcement Learning Computation Flow Overview

1. **Rollout**: The policy (actor) model generates responses (response sequences) through inference based on the input prompt sequence.
2. **ref logprob**: Based on the prompt and the generated response, the reference model computes the ref logprob for KL divergence calculation.
3. **logprob**: Based on the prompt and the generated response, the actor model computes the logprob for importance sampling.
4. **reward**: Based on the prompt and the generated response, the reward model evaluates the reward value R_N.
5. **update**: Based on the computed R_N, ref logprob, and logprob, the system computes the optimization function and policy gradient to update the actor model.

![rl_data_stream](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/rl_data_stream.png)

## Enabling the profiling tool

### Enabling Method

For enabling and configuration tutorials, refer to the [Profiling Collection Guide](./ascend_profiling.rst)

## Performance Analysis Methodology

### Overall Performance Overview Analysis

#### 1. Long-Running Task and Resource Idle Period Analysis

- **Operation**: Use MindStudio Insight to load profiling data, automatically identify different computation stages, and locate long-duration tasks and NPU resource bubbles through the RL tab pipeline graph
- **Value**: Quickly understand the time consumption proportion of different stages
- **Demonstration**:

![Bubble_analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Bubble_analysis.png)

#### 2. Load balancing analysis

- **Operation**: View MSTX trace data directly using MindStudio Insight to observe the load balancing status of different DP Ranks during the Rollout phase
- **Value**: Quickly identify load imbalance issues
- **Result display:**

![Load_Balancing_Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Load_Balancing_Analysis.gif)

#### 3. Overall Cluster Performance Analysis

- **Operation**: Using the rl_analysis feature of MSTT, generate a cluster Timeline thumbnail to observe the overall duration of each stage
- **Value**: Gain a macro-level understanding of cluster performance bottlenecks
- **User guide**: [rl_analysis documentation](https://gitcode.com/Ascend/mstt/blob/pre-research/profiler/msprof_analyze/docs/features/rl_analysis.md)
- **Result demonstration**:

![Cluster%20Performance%20Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Cluster%20Performance%20Analysis.png)

### Fine-grained analysis

#### Performance Analysis

- **Operation**: You can load Profiling data using the MindStudio Insight Windows or Linux version.
- **Value**: MindStudio Insight supports analyzing task scheduling efficiency, operator execution performance, computing resource utilization, collective communication performance, and so on. Its Timeline view provides task decomposition and Overlap analysis functions (**a core feature unique to MindStudio, not available in NV and other competing products, and an essential tool for AI tuning**), and supports interactive mouse analysis.
- **Result display**:

![performance%20analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/performance%20analysis.png)

#### Memory Analysis

##### **Analyzing system memory changes using Profiling and call stack analysis**

- **Operation**: Enable the call stack and memory view functions when collecting data.
- **Value**: Observe the memory allocation and release status of the framework and CANN. You can use the call stack to trace back to the frontend Python code.
- **Effect display**: Analyze memory changes combined with the call stack. The effect is as follows:

![in-memory%20analytics](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/in-memory%20analytics.gif)

##### **Performing in-depth memory analysis using the msleaks tool**

- **Procedure**: Refer to the [msleaks Tool Usage Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/msMemScope/docs/en/memory_analysis.md).
- **Value**: You can view the total framework memory allocation line chart and memory block diagrams, and directly map them to call stacks for in-depth analysis of framework memory usage.
- **Result display**:

![msleaks](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/msleaks.gif)

## Performance Analysis Cases

To perform detailed performance analysis, enable **level1** profiling; otherwise, key operator information is missing.

### 1. Host-bound diagnosis

Host bound refers to the phenomenon where the total CPU task volume exceeds that of the NPU, causing bubbles in NPU execution. You can determine this by checking the Host2Device synchronization lines. If the lines are skewed, it indicates that the set signal occurs earlier than the wait signal. The NPU executes as soon as it is ready. This is also device bound:

![host_bound_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_1.png)

If you diagnose it as host bound, you can enable the CPU side to find the dispatch time of each operator. Note that you need to find the accumulated value of all CPU time instead of a single layer. This is because the first call takes a long time. For example, for GmmSwigluQuant in the following figure, the first call on the CPU takes 1 ms. Each subsequent call takes only 200 μs.

![host_bound_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_2.png)

At this point, some operators carry the heavy workload while others cause bottlenecks, and the latter outnumber the former. Prioritize **identifying the top operators whose host time is greater than the device time; these operators are the bottlenecks**. You can assign them to the operator team for focused analysis.

### 2. Networking Rationality Analysis

Sometimes, the model network is not constructed in the most efficient way. This is easy to identify during profiling. The following section introduces the analysis approach and provides examples.

Generally, the major hot operators in LLMs are the matrix multiplication computations in Attention and FFN. Together, they may account for 70%+ of the computation time during prefill and 50%+ during decode. If the overall time proportion does not meet expectations, or unfamiliar operators appear in the profiling results, or there are too many concatenation-type operators, analyze the model architecture to check whether the operators are used incorrectly. Concatenation-type operators in particular are worth analyzing one by one.

Concatenation operators (such as slice, split, and concat) and conversion operators (such as transpose and cast) often exist because the preceding and following operators are not directly compatible. If the preceding operator can directly perform post-processing on the output, it can save the startup overhead of one operator and one redundant read/write. However, such a change may not conform to the basic design principles of operators.

As a positive example, consider a Matmul operation with an output shape of [m, n0 + n1]. We connect two slice operations after it, both taking the [m, n0 + n1] tensor as input, producing outputs of [m, n0] and [m, n1] respectively. The first optimization approach is to replace the two slice operations with a single split. This roughly halves the time consumption and allows the device memory of [m, n0 + n1] to be released as early as possible. A further optimization approach is to split the matrix multiplication weight from [k, n0 + n1] into [k, n0] and [k, n1], dividing the original matrix multiplication task into two. This completely eliminates the slice/split operation, provided that the combined time consumption of the two does not degrade significantly, and the core partitioning strategy works correctly.

![network_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_1.png)

Consider a counterexample: Rmsnorm(fp16)+Cast(fp16->fp32)+Matmul(fp32). Although the input and output of Rmsnorm are both fp16, it uses fp32 for internal computation to ensure the precision of accumulation operations. If the Cast is fused into Rmsnorm, the Rmsnorm, which already uses fp32 for internal computation, can eliminate a trailing fp32-to-fp16 cast. Combined with the Cast that we remove, this saves two casts in total while avoiding one precision loss. Although this appears to benefit both precision and performance, an Rmsnorm that takes fp16 as input and produces fp32 as output violates a core principle. This principle requires that the core input and output must be of the same data type. Unless we can frequently find this structure in open-source models to prove its universality, the operator team will not allow creating such an operator.

![network_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_2.png)

### 3. Preliminary Diagnosis of Operator Performance

Use `"./ASCEND_PROFILER_OUTPUT/operator_details.csv"` to analyze whether the operator has performance issues.

The Profiling tool collects the average busy time of these pipelines on different cores (xxx_time), divides it by the complete kernel duration of the slowest core (task_duration), and obtains the pipeline utilization (xxx_ratio). Although these pipelines have dependencies on each other, and data movement pipelines compete for bandwidth, they can overlap with each other if the operator is designed properly. Therefore, we can preliminarily conclude that **when the execution duration of an operator reaches a certain level, the operator should form a bound on one of the pipelines**, that is, the utilization should reach a certain level. Based on experience, when the single-operator duration reaches 50μs, it can be considered that the operator should be on the bound pipeline, achieving an occupancy rate of 80%+.

Take the following figure as an example. The first row is an FA operator, and the second row is a Matmul operator. The FA operator achieves a utilization rate of 88.1% on the vec pipeline. The Matmul operator achieves a utilization rate of 89.8% on the mac pipeline. Their performance is considered acceptable.

![Operator%20performance](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Operator%20performance.png)

### 4. Affinity shape adjustment

For a model, hyperparameters are beyond your control. However, you can control factors such as concurrency, weight format, and sharding strategy to accommodate the operators and maximize their performance. This section discusses adjustment directions worth trying on the model side, focusing on operator data movement efficiency and load balancing.

#### 4.1 Data movement efficiency-friendly shapes

mte2 is a pipeline whose efficiency is severely affected by shape. To ensure that mte2 achieves maximum transfer efficiency, ensure that at least one of the following two conditions is met:

**（1）The matrix being moved uses nz as the format (optimal)
（2）The last axis of the matrix being moved is 512B aligned and is not an integer multiple of 16KB (near-optimal)**

For weight matrices, during the inference phase, especially during decode, we typically satisfy condition (1), and during the training phase, we typically satisfy condition (2). **If we cannot satisfy (1), we must accommodate (2)**. Typical approaches include:

1. If condition (1) is not met, and the leading axis of the matrix is contiguous while the trailing axis is not, transpose it.
2. Adjust the TP sharding strategy to avoid non-contiguous trailing axes.

#### 4.2 Load balancing affinity shape

When the operator shape is small, you might not use all cores due to operator semantics. Even if you enable all cores, the load balancing might be poor. This subsection mainly analyzes small shapes in the decode phase.

First, determine the number of cores on the current NPU card. If you are unsure, check the profiling results. If the results show numbers such as 20 or 40, the card has 20 cores. Otherwise, the card has 24 cores. Here, the 24 cores actually represent a group consisting of one cube and two vectors. You can consider one cube as the primary core and two vectors as secondary cores. If an operator is a pure vector operator, the concept of a group no longer applies. The 40 or 48 vector cores act as primary cores and independently fetch logical tasks.

For vector operators in LLMs, one common core partitioning strategy is to partition along the highest dimension, that is, the batch dimension. This strategy is common for operators that perform reduction operations on low dimensions (also known as tail axes), such as normalization and dynamic quantization operators. Another strategy is to flatten the data as a whole, which allows operators to be split into very fine granularity, such as elementwise operators. For the first strategy, you can focus on the load balancing issue on the model side. For example, if you set the batch size to 48 and the hardware has 40 vector cores, the 40 cores loop twice. In the second loop, most cores remain idle. This batch size can be considered unfriendly. If you set the batch size to 64 or 80, the performance is expected to be lossless. Under the same conditions, if the card has 48 cores, this batch size can be considered very friendly.

For cube operators, a common core partitioning strategy is to split the M and N axes by using base blocks. The K axis is the accumulation axis, and partitioning it introduces determinism issues. The most common block sizes are baseM=128 and baseN=256. During the decode phase, the time consumption is primarily spent moving weights. The M dimension of the activation is extremely small, so the M direction is likely split into only one block, and the right matrix needs to be moved only once. Therefore, you can increase M freely within the range of M ≤ 128, which is basically lossless for performance. If M is greater than 128, consider the range (128, 256] as the next performance tier. In addition to M, the task partitioning along the N axis also affects operator affinity. For example, the MLA preprocessing in deepseekR1 uses the same activation (with a shape of [batch_size, 7168]) to perform matrix multiplication with two weights (with shapes of [7168, 1536] and [7168, 576]). When the batch_size is small, even if baseN is reduced to 128, the N axis cannot fully utilize all cores. Therefore, the time consumption of each of these two matrix multiplications approximately equals the time consumption of a single matrix multiplication (with a shape of [7168, 2112]) that concatenates the N axes of the two weights. If you only consider model competitiveness, it is preferable to merge these two weights. Otherwise, the bandwidth utilization of both small matrix multiplications will be very poor.

For the Attention operator, the common core partitioning strategy involves q_seqlen, batch_size, and kv_headnum. During the incremental phase, q_seqlen is merged by the MTP and GQA multiples, but it typically does not exceed 128. Because a second task cannot be partitioned, the parallelism is basically batch_size * kv_headnum.

In summary, based on the shape information and operator category, you can identify whether the operator has a load balancing problem. This helps you predict the split strategy selection and the batch strategy for maximum throughput.
