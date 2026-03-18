Performance Tuning Guide on Ascend
====================================

Last updated:  01/29/2026.

Author:  `Xiaobo Hu <https: //github.com/tardis-key>`_, `Haozhe Li <https: //github.com/ZLiao097>`_

`Perf Tuning <https: //github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ 中介绍的性能调优方法在昇腾设备中同样适用。本文重点介绍了昇腾特有的一些调优手段，包括融合算子优化、特定硬件配置和昇腾亲和特性等。

融合算子
--------------------------

常用融合算子列表
**********************************

融合算子的优化原理为，通过数学意义上的等价替换，将多个算子融为一个算子的计算，减少冗余计算，同时减少下发次数，从而提高性能。几个典型的NPU融合算子列举如下，目前均已在 npu_patch.py 中对 Qwen2、Qwen3 系列模型完成替换。

当前verl中使用的全量融合算子请查阅 `npu_patch.py <https: //github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_ 

Matrix Computation-Communication operator fusion (MC2) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MC2 是 CANN 中一系列计算通信融合算子的统称，这些算子将原本串行的通信和计算操作融合在一起，通过内部的切分和流水线并行执行来优化性能。

在 vllm-ascend 中，可以通过指定环境变量：

.. code-block:: sh

    export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1

在前向计算的 ``RowParallelLinear`` 中使能 ``torch_npu.npu_mm_all_reduce_base`` ，将分离的 ``matmul`` 和 ``allreduce`` 合并为一个融合算子。

`RotaryMul&RotaryMulGrad <https: //www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0030.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu 接口:  ``torch_npu.npu_rotary_mul(x, r1, r2)``

参数说明: 

- x: q，k，shape要求输入为4维，一般为 ``[B, N, S, D]`` 或 ``[B, S, N, D]`` 或 ``[S, B, N, D]`` 。

- r1: cos值 ，shape要求输入为4维，一般为 ``[1, 1, S, D]`` 或 ``[1, S, 1, D]`` 或 ``[S, 1, 1, D]`` 。

- r2: sin 值，shape要求输入为4维，一般为 ``[1, 1, S, D]`` 或 ``[1, S, 1, D]`` 或 ``[S, 1, 1, D]`` 。

`RmsNorm&RmsNormGrad <https: //www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0031.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu 接口:  ``torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor)`` 
参数说明: 

- self: Tensor 类型，shape 支持 1-8 维。

- gamma: Tensor 类型，通常为weight，shape 要求与 self 的后几维保持一致。

- epsilon: Float 数据类型，用于防止除 0 错误。

输出说明: 

- 第 1 个输出为 Tensor，计算公式的最终输出y。

- 第 2 个输出为 Tensor， rms_norm 的中间结果 rstd ，用于反向计算。

`Swiglu <https: //www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0035.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu 接口:  ``torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)`` 

参数说明: 

- self: Tensor 类型，shape支持 1-8 维。

- dim: Int 类型，默认为 -1。

输出说明: 

- 输出为 Tensor，计算公式的最终输出 y。

`GroupMatMul <https: //www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_grouped_matmul.md>`_
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

详细使用方法见标题文档链接

FSDP后端融合算子使用方法
**********************************

在 ``verl/models/transformers/npu_patch.py`` 目录中，已经把可用的融合算子通过 patch 的形式进行替换，无需进行其他操作即可默认进行使用

Megatron后端融合算子使用方法
**********************************

Megatron 的融合算子集成在 MindSpeed 中，需要添加特定参数开启: 

1. **Flash Attention（必须开启）**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
       ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True

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

5. **Swiglu**
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

`算子下发 <https: //www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
************************************************************************************************************************************************************************************************************

通过 ``TASK_QUEUE_ENABLE`` 可配置 task_queue 算子下发队列优化等级，默认为 Level 1 优化。该配置可以减少host下发时间，可用于缓解由下发导致的整体free过大问题。

.. image :: https://github.com/verl-project/verl-data/blob/main/images/ascend/perf_tuning_task_queue.png
    :width: 500px

Level 0 : 不开启下发流水优化。

Level 1 : \ 将算子下发任务分为两段，一部分任务（主要是 aclnn 算子的调用）放在新增的二级流水上，一、二级流水通过算子队列传递任务，相互并行，通过部分掩盖减少整体的下发耗时，提升端到端性能。

Level 2 : \ 基于 Level 1 的优化进一步平衡了一、二级流水的任务负载，主要是将 workspace 相关任务迁移至二级流水，掩盖效果更好，性能收益更大。该配置仅在二进制场景生效，建议配置值为 Level 2 优化。

`通讯算法编排展开 <https: //www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
************************************************************************************************************************************************************************************************************
使用环境变量 ``HCCL_OP_EXPANSION_MODE=AIV`` 用于配置通信算法的编排展开位置，支持如下取值: 

- **AI_CPU:** 代表通信算法的编排展开位置在 Device 侧的 AI CPU，Device 侧根据硬件型号自动选择相应的调度器。

- **AIV:** 代表通信算法的编排展开位置在 Device 侧的 Vector Core，执行也在 Vector Core。

- **HOST:** 代表通信算法的编排展开位置为 Host 侧 CPU，Device 侧根据硬件型号自动选择相应的调度器。

- **HOST_TS:** 代表通信算法的编排展开位置为 Host 侧 CPU，Host 向 Device 的 Task Scheduler 下发任务，Device 的 Task Scheduler 进行任务调度执行。

推理阶段调优
--------------------------

Chunked Prefill in V1
***************************

VLLM 当前版本已默认启用 VLLM V1，使用以下配置启用 Chunked Prefill：

.. code-block:: sh

    actor_rollout_ref.rollout.enable_chunked_prefill=True

原理参考 `VLLM 官方文档 <https://docs.vllm.ai/en/v0.4.2/models/performance.html>`_。

Graph Mode
***************************

与 CUDA 类似，NPU 通过以下配置启用 **ACL Graph**：

.. code-block:: sh

    actor_rollout_ref.rollout.enforce_eager=False

文档：`ACL Graph <https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/feature_guide/ACL_Graph.html>`_

.. note::
    ACL Graph 与 ``taskqueue Level 2`` 原理冲突，**二者无法同时开启**。

训练阶段调优
--------------------------

FSDP
**********************************

.. csv-table::
   :header: "FSDP", "说明"
   :widths: 30, 60

   "/","仅切分优化器(Zero-1)"
   SHARD_GRAD_OP,切分梯度和优化器(Zero-2)
   "HYBRID_SHARD","切分权重、梯度和优化器(Zero-3)"
   "2D device_mesh+HYBRID_SHARD","又称HSDP（FSDP+DDP）例如device_mesh=[2,8], 每8个rank为一个FSDP组，组内进行FSDP切分，共有两个组，两个组间进行DDP，通过allreduce同步梯度。"
   "2D device_mesh+HYBRID_SHARD_ZERO2","HSDP的Zero2版本"
   NO_SHARD,DDP

FSDP 不支持 Zero-1， VeRL中会根据卡数和 ``actor_rollout_ref.actor.fsdp_config.fsdp_size``  来决定 device mesh 的取值，默认使用 Zero-3 进行切分；如果模型较小（建议小于 7B 时），可以通过控制参数 ``actor_rollout_ref.actor.fsdp_config.reshard_after_forward`` 为 ``True`` 在 FSDP/FSDP2 上使用 Zero-2 来优化性能.

Megatron
**********************************

在模型较大时，使用 Megatron 作为训练后端可以更灵活的进行性能调优。

当 DP 并行显存无法容纳模型时，优先开启 TP 来切分模型权重，如果模型仍然过大，再开启 PP 来进一步切分；如果序列过长导致激活太大，则可以开启 CP 和 SP 来进行优化；在 MoE 模型中则可以额外开启 EP 来控制对专家的切分，如果专家过小，为了避免将权重切的果味细碎，则可以开启 ETP 来避免 MoE 部分的 TP 切分，而将多个完整的专家分布到 DP 和 TP 上。

TP、PP、EP、ETP和 Megatron 使用方式一样，CP 和 SP 在 NPU 上开启方式: 

- SP: ``Sequence Parallel`` 在 Tensor Parallel 的基础上进一步提高计算效率，是一种通过将输入数据的序列维度进行切分的并行计算方式。在 NPU 上通过 MindSpeed 来调用SP:
  ::

      actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True

- CP: ``Context Parallel`` 是一种在多个 GPU/NPU 上并行处理神经网络激活值的方法，他通过在序列维度上对输入张量进行划分来实现。在 NPU 上通过 MindSpeed 来调用 CP （两个参数必须同时添加）:
  ::

      actor_rollout_ref.actor.megatron.context_parallel_size
      actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
**********************************

在面对较大尺寸模型时，通常需要将优化器分片到一个 DP 域内的每张卡上来节省显存。Megatron 后端下在 NPU 上开启分布式优化器:

::

    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
