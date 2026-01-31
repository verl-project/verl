Performance Tuning Guide on Ascend
==============================

Last updated: 01/29/2026.

Author: `Xiaobo Hu <https://github.com/tardis-key>`_, `Haozhe Li <https://github.com/ZLiao097>`_

`Perf Tuning <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ 中介绍的性能调优方法在昇腾设备中同样适用。本文重点介绍了昇腾特有的一些调优手段，包括融合算子优化、特定硬件配置和昇腾亲和特性等。

融合算子
--------------------------

常用融合算子列表
""""""""""""""""""""""

融合算子的优化原理为，通过数学意义上的等价替换，将多个算子融为一个算子的计算，减少冗余计算，同时减少下发次数，从而提高性能。几个典型的NPU融合算子列举如下，目前均已在 npu_patch.py 中对Qwen2、Qwen3系列模型完成替换。

当前verl中使用的全量融合算子请查阅`npu patch <https://github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_ 

Matrix Computation-Communication operator fusion (MC2)
MC2是CANN中一系列计算通信融合算子的统称，这些算子将原本串行的通信和计算操作融合在一起，通过内部的切分和流水线并行执行来优化性能。在 vllm-ascend 中，可以通过指定环境变量VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1 ，在前向计算的RowParallelLinear中使能torch_npu.npu_mm_all_reduce_base，将分离的mat mul和allreduce合并为一个融合算子。

`RotaryMul&RotaryMulGrad <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0030.html>`_

torch_npu接口：torch_npu.npu_rotary_mul(x, r1, r2)
参数说明：
- x：q，k，shape要求输入为4维，一般为[B, N, S, D]或[B, S, N, D]或[S, B, N, D]。
- r1：cos值，shape要求输入为4维，一般为[1, 1, S, D]或[1, S, 1, D]或[S, 1, 1, D]。
- r2：sin值，shape要求输入为4维，一般为[1, 1, S, D]或[1, S, 1, D]或[S, 1, 1, D]。

`RmsNorm&RmsNormGrad <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0031.html>`_

torch_npu接口：torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor)
参数说明：
- self：Tensor类型，shape支持1-8维。
- gamma：Tensor类型，通常为weight，shape要求与self的后几维保持一致。
- epsilon：float数据类型，用于防止除0错误。
输出说明：
- 第1个输出为Tensor，计算公式的最终输出y。
- 第2个输出为Tensor，rms_norm的中间结果rstd，用于反向计算。

`Swiglu <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0035.html>`_

torch_npu接口：torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)
参数说明：
- self：Tensor类型，shape支持1-8维。
- dim：int类型，默认为-1。
输出说明：
- 输出为Tensor，计算公式的最终输出y。

`GroupMatMul <https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_grouped_matmul.md>`_

fsdp后段融合算子使能
""""""""""""""""""""""

在verl/models/transformers/npu_patch.py目录中，已经把可用的融合算子通过patch的形式进行替换，无需进行其他操作即可默认进行使用

megatron后段融合算子使能
""""""""""""""""""""""

megatron的融合算子集成在mindspeed中，需要添加特定参数开启
Flash Attention（必须开启）：
+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True
RotaryMul：
+actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
RMSNorm：
+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True
GroupMatMul：
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
Swiglu:
+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True
Permute/Unpermute：
+actor_rollout_ref.actor.megatron.override_transformer_config.fused_permute_unpermute=True
MC2:
+actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2

昇腾通用配置
--------------------------

`算子下发 <https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
""""""""""""""""""""""

通过TASK_QUEUE_ENABLE可配置task_queue算子下发队列是否开启和优化等级，默认为 Level1优化
Level 1优化：使能task_queue算子下发队列优化，将算子下发任务分为两段，一部分任务（主要是aclnn算子的调用）放在新增的二级流水上，一、二级流水通过算子队列传递任务，相互并行，通过部分掩盖减少整体的下发耗时，提升端到端性能。
Level 2优化：包含Level 1的优化并进一步平衡了一、二级流水的任务负载，主要是将workspace相关任务迁移至二级流水，掩盖效果更好，性能收益更大。该配置仅在二进制场景生效，建议配置值为Level 2优化。

`通讯算法编排展开 <https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
""""""""""""""""""""""

HCCL_OP_EXPANSION_MODE=AIV
该环境变量用于配置通信算法的编排展开位置，支持如下取值：
- AI_CPU：代表通信算法的编排展开位置在Device侧的AI CPU，Device侧根据硬件型号自动选择相应的调度器。
- AIV：代表通信算法的编排展开位置在Device侧的Vector Core，执行也在Vector Core。
- HOST：代表通信算法的编排展开位置为Host侧CPU，Device侧根据硬件型号自动选择相应的调度器。
- HOST_TS：代表通信算法的编排展开位置为Host侧CPU，Host向Device的Task Scheduler下发任务，Device的Task Scheduler进行任务调度执行。

推理阶段调优
--------------------------

Chunked Prefill in V1
""""""""""""""""""""""
Vllm 当前版本已默认启用vllm V1，使用 actor_rollout_ref.rollout.enable_chunked_prefill=True 来启用Chunked Prefill，原理参考 vllm 官方文档

Graph Mode 
""""""""""""""""""""""

与cuda一样，NPU通过actor_rollout_ref.rollout.enforce_eager=False来启用图模式，注意由于其原理与taskqueue level2存在冲突，二者无法同时开启

训练阶段调优
--------------------------

fsdp
""""""""""""""""""""""

表格todo
FSDP不支持Zero-1，VeRL中会根据卡数和 actor_rollout_ref.actor.fsdp_config.fsdp_size 来决定device mesh，默认使用 Zero-3 进行切分，如果模型较小（通常建议小于7B时），可以更改参数 actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False 在FSDP/FSDP2上使用Zero-2来优化性能

Megatron
""""""""""""""""""""""

在模型较大时，使用megatron作为训练后端可以更灵活的进行性能调优。
当DP并行显存无法容纳模型时，优先开启TP来切分模型权重，如果模型仍然过大，再开启PP来进一步切分；如果序列过长导致激活太大，则可以开启CP和SP来进行优化；在MoE模型中则可以额外开启EP来控制对专家的切分，如果专家过小，为了避免将权重切的果味细碎，则可以开启ETP来避免MoE部分的TP切分，而将多个完整的专家分布到DP和TP上。
TP、PP、EP、ETP和megatron使用方式一样，CP和SP在NPU上开启方式：
- SP：Sequence Parallel 在 Tensor Parallel的基础上进一步提高计算效率，是一种通过将输入数据的序列维度进行切分的并行计算方式。在NPU上通过mindspeed来调用SP：actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
- CP：Context Parallel 是一种在多个GPU上并行处理神经网络激活值的方法，他通过在序列维度上对输入张量进行划分来实现。在NPU上通过mindspeed来调用CP（两个参数需要同时添加）：
actor_rollout_ref.actor.megatron.context_parallel_size
actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
""""""""""""""""""""""

在使用megatron后端时，面对较大尺寸模型通常需要开启分布式优化器来将优化器分片到一个DP域内的每张卡上。在NPU上开启分布式优化器：
+actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
