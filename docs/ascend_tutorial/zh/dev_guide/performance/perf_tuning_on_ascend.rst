昇腾性能调优指南
====================================

Last updated: 01/29/2026.

Author: `Xiaobo Hu <https://github.com/tardis-key>`_, `Haozhe Li <https://github.com/ZLiao097>`_

`Perf Tuning <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ 中介绍的性能调优方法在昇腾设备中同样适用。本文重点介绍了昇腾特有的一些调优手段，包括融合算子优化、特定硬件配置和昇腾亲和特性等。

融合算子
--------------------------

常用融合算子列表
**********************************

融合算子的优化原理为，通过数学意义上的等价替换，将多个算子融合为一个算子的计算，减少冗余计算，同时减少下发次数，从而提高性能。几个典型的NPU融合算子列举如下，目前均已在npu_patch.py中对Qwen2、Qwen3系列模型完成替换。

当前verl中使用的全量融合算子请查阅 `npu_patch.py <https://github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_ 

Matrix Computation-Communication operator fusion (MC2) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MC2是CANN中一系列计算通信融合算子的统称，这些算子将原本串行的通信和计算操作融合在一起，通过内部的切分和流水线并行执行来优化性能。

在vllm-ascend中，可以通过指定环境变量：

.. code-block:: sh

    export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1

在前向计算的 ``RowParallelLinear`` 中使能 ``torch_npu.npu_mm_all_reduce_base``，将分离的 ``matmul`` 和 ``allreduce`` 合并为一个融合算子。

`RotaryMul&RotaryMulGrad <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu接口： ``torch_npu.npu_rotary_mul(x, r1, r2)``

参数说明： 

- x: q，k，shape要求输入为4维，一般为 ``[B, N, S, D]`` 或 ``[B, S, N, D]`` 或 ``[S, B, N, D]``。

- r1: cos值，shape要求输入为4维，一般为 ``[1, 1, S, D]`` 或 ``[1, S, 1, D]`` 或 ``[S, 1, 1, D]``。

- r2: sin值，shape要求输入为4维，一般为 ``[1, 1, S, D]`` 或 ``[1, S, 1, D]`` 或 ``[S, 1, 1, D]``。

`RmsNorm&RmsNormGrad <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu接口: ``torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor)`` 
参数说明: 

- self: Tensor类型，shape支持1-8维。

- gamma: Tensor类型，通常为weight，shape要求与self的后几维保持一致。

- epsilon: Float数据类型，用于防止除0错误。

输出说明：

- 第1个输出为Tensor，计算公式的最终输出y。

- 第2个输出为Tensor，rms_norm的中间结果rstd，用于反向计算。

`SwiGLU <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_swiglu.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu接口： ``torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)`` 

参数说明： 

- self: Tensor类型，shape支持1-8维。

- dim: Int类型，默认为 -1。

输出说明： 

- 输出为Tensor，计算公式的最终输出y。

`GroupMatMul <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

函数原型：

.. code:: python

    npu_grouped_matmul(
        x, 
        weight, 
        *, 
        bias=None, 
        scale=None, 
        offset=None, 
        antiquant_scale=None, 
        antiquant_offset=None, 
        per_token_scale=None, 
        group_list=None, 
        activation_input=None, 
        activation_quant_scale=None, 
        activation_quant_offset=None, 
        split_item=0, group_type=None, 
        group_list_type=0, 
        act_type=0, 
        output_dtype=None, 
        tuning_config=None
    ) -> List[Tensor]

详细使用方法见标题中的文档链接。

FSDP后端融合算子使用方法
**********************************

在 ``verl/models/transformers/npu_patch.py`` 文件中，已经把可用的融合算子通过patch的形式进行替换，无需进行其他操作即可默认进行使用。

Megatron后端融合算子使用方法
**********************************

Megatron的融合算子集成在MindSpeed中，需要添加特定参数开启: 

1. **Flash Attention（必须开启）**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True

2. **RotaryMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True

3. **RMSNorm**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True

4. **GroupMatMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True

5. **SwiGLU**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True

6. **Permute/Unpermute**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.fused_permute_unpermute=True

7. **MC2**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2=True

昇腾通用配置
--------------------------

`算子下发 <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/ENV/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
************************************************************************************************************************************************************************************************************

通过 ``TASK_QUEUE_ENABLE`` 可配置task_queue算子下发队列优化等级，默认为Level 1优化。该配置可以减少Host下发时间，可用于缓解由下发导致的整体free过大问题。

.. image :: https://github.com/verl-project/verl-data/blob/main/images/ascend/perf_tuning_task_queue.png
    :alt: TASK_QUEUE_ENABLE不同下发优化等级的任务队列流水示意图，展示一二级流水并行掩盖下发耗时
    :width: 500px

Level 0 : 不开启下发流水优化。

Level 1 : 将算子下发任务分为两段，一部分任务（主要是aclnn算子的调用）放在新增的二级流水上，一、二级流水通过算子队列传递任务，相互并行，通过部分掩盖减少整体的下发耗时，提升端到端性能。

Level 2 : 基于Level 1的优化进一步平衡了一、二级流水的任务负载，主要是将workspace相关任务迁移至二级流水，掩盖效果更好，性能收益更大。该配置仅在二进制场景生效，建议配置值为Level 2优化。

`通信算法编排展开 <https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
************************************************************************************************************************************************************************************************************
使用环境变量 ``HCCL_OP_EXPANSION_MODE=AIV`` 用于配置通信算法的编排展开位置，支持如下取值: 

- **AI_CPU:** 代表通信算法的编排展开位置在Device侧的AI CPU，Device侧根据硬件型号自动选择相应的调度器。

- **AIV:** 代表通信算法的编排展开位置在Device侧的Vector Core，执行也在Vector Core。

- **HOST:** 代表通信算法的编排展开位置为Host侧CPU，Device侧根据硬件型号自动选择相应的调度器。

- **HOST_TS:** 代表通信算法的编排展开位置为Host侧CPU，Host向Device的Task Scheduler下发任务，Device的Task Scheduler进行任务调度执行。

推理阶段调优
--------------------------

Chunked Prefill in V1
***************************

VLLM当前版本已默认启用VLLM V1，使用以下配置启用Chunked Prefill：

.. code-block:: sh

    actor_rollout_ref.rollout.enable_chunked_prefill=True

原理参考 `VLLM 官方文档 <https://docs.vllm.ai/en/v0.4.2/models/performance.html>`_。

Graph Mode
***************************

与CUDA类似，NPU通过以下配置启用 **ACL Graph**：

.. code-block:: sh

    actor_rollout_ref.rollout.enforce_eager=False

.. note::
    ACL Graph与 ``taskqueue Level 2`` 原理冲突，**二者无法同时开启**。

训练阶段调优
--------------------------

FSDP
**********************************

.. csv-table::
   :header: "FSDP", "说明"
   :widths: 30, 60

   "/","仅切分优化器(ZeRO-1)"
   SHARD_GRAD_OP,切分梯度和优化器(ZeRO-2)
   "HYBRID_SHARD","切分权重、梯度和优化器(ZeRO-3)"
   "2D device_mesh+HYBRID_SHARD","又称HSDP（FSDP+DDP）例如device_mesh=[2,8]，每8个rank为一个FSDP组，组内进行FSDP切分，共有两个组，两个组间进行DDP，通过allreduce同步梯度。"
   "2D device_mesh+HYBRID_SHARD_ZERO2","HSDP的Zero2版本"
   NO_SHARD,DDP

FSDP不支持ZeRO-1，verl中会根据卡数和 ``actor_rollout_ref.actor.fsdp_config.fsdp_size``  来决定device mesh的取值，默认使用ZeRO-3进行切分；如果模型较小（建议小于7B时），可以通过控制参数 ``actor_rollout_ref.actor.fsdp_config.reshard_after_forward`` 为 ``True`` 在FSDP/FSDP2上使用ZeRO-2来优化性能。

Megatron
**********************************

在模型较大时，使用Megatron作为训练后端可以更灵活地进行性能调优。

当DP并行显存无法容纳模型时，优先开启TP来切分模型权重，如果模型仍然过大，再开启PP来进一步切分；如果序列过长导致激活太大，则可以开启CP和SP来进行优化；在MoE模型中则可以额外开启EP来控制对专家的切分，如果专家过小，为了避免将权重切得过于细碎，则可以开启ETP来避免MoE部分的TP切分，而将多个完整的专家分布到DP和TP上。

TP、PP、EP、ETP和Megatron使用方式一样，CP和SP在NPU上开启方式: 

- SP: ``Sequence Parallel`` 在Tensor Parallel的基础上进一步提高计算效率，是一种通过将输入数据的序列维度进行切分的并行计算方式。在NPU上通过MindSpeed来调用SP:
  ::

      actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True

- CP: ``Context Parallel`` 是一种在多个GPU/NPU上并行处理神经网络激活值的方法，它通过在序列维度上对输入张量进行划分来实现。在NPU上通过MindSpeed来调用CP（两个参数必须同时添加）:
  ::

      actor_rollout_ref.actor.megatron.context_parallel_size
      actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
**********************************

在面对较大尺寸模型时，通常需要将优化器分片到一个DP域内的每张卡上来节省显存。Megatron后端下在NPU上开启分布式优化器:

::

    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
