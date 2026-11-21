Ascend Performance Tuning Guide
====================================

Last updated:  01/29/2026.

Author:  `Xiaobo Hu <https://github.com/tardis-key>`_, `Haozhe Li <https://github.com/ZLiao097>`_

The performance tuning methods introduced in `Perf Tuning <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ also apply to Ascend devices. This document focuses on several Ascend-specific tuning approaches, including fusion operator optimization, specific hardware configurations, and Ascend affinity features.

Fusion Operators
--------------------------

List of Commonly Used Fusion Operators
**********************************

The optimization principle of fusion operators is to use mathematically equivalent substitutions to fuse the computation of multiple operators into a single operator, reducing redundant computation and the number of dispatches, thereby improving performance. Several typical NPU fusion operators are listed below; all of them have already been substituted for the Qwen2 and Qwen3 series models in npu_patch.py.

For the full set of fusion operators currently used in verl, refer to `npu_patch.py <https://github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_ 

Matrix Computation-Communication operator fusion (MC2) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MC2 is a general term for a series of compute-communication fusion operators in CANN. These operators fuse the originally serial communication and computation operations together, optimizing performance through internal splitting and pipelined parallel execution.

In vllm-ascend, you can specify the environment variable:

.. code-block:: sh

    export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1

This enables ``torch_npu.npu_mm_all_reduce_base`` in the ``RowParallelLinear`` of forward computation, merging the separate ``matmul`` and ``allreduce`` into a single fusion operator.

`RotaryMul&RotaryMulGrad <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface:  ``torch_npu.npu_rotary_mul(x, r1, r2)``

Parameter description: 

- x: q, k. The shape requires 4-dimensional input, generally ``[B, N, S, D]`` or ``[B, S, N, D]`` or ``[S, B, N, D]``.

- r1: cos value. The shape requires 4-dimensional input, generally ``[1, 1, S, D]`` or ``[1, S, 1, D]`` or ``[S, 1, 1, D]``.

- r2: sin value. The shape requires 4-dimensional input, generally ``[1, 1, S, D]`` or ``[1, S, 1, D]`` or ``[S, 1, 1, D]``.

`RmsNorm&RmsNormGrad <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/torch_npu/(beta)torch_npu-npu_rms_norm.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface:  ``torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor)`` 
Parameter description: 

- self: Tensor type, shape supports 1-8 dimensions.

- gamma: Tensor type, usually the weight. The shape must match the last few dimensions of self.

- epsilon: Float data type, used to prevent divide-by-zero errors.

Output description:

- The 1st output is a Tensor, the final output y of the formula.

- The 2nd output is a Tensor, the intermediate result rstd of rms_norm, used for backward computation.

`Swiglu <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/torch_npu/(beta)torch_npu-npu_swiglu.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface:  ``torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)`` 

Parameter description: 

- self: Tensor type, shape supports 1-8 dimensions.

- dim: Int type, default is -1.

Output description: 

- The output is a Tensor, the final output y of the formula.

`GroupMatMul <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function prototype:

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

For detailed usage, refer to the linked title documentation.

Using Fusion Operators with the FSDP Backend
**********************************

In ``verl/models/transformers/npu_patch.py``, the available fusion operators have already been substituted through patching. No additional operations are required; they are used by default.

Using Fusion Operators with the Megatron Backend
**********************************

Megatron's fusion operators are integrated in MindSpeed and require specific parameters to enable: 

1. **Flash Attention (must be enabled)**
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

5. **Swiglu**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True

6. **Permute/Unpermute**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.fused_permute_unpermute=True

7. **MC2**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2=True

Ascend General Configuration
--------------------------

`Operator Dispatch <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/ENV/docs/en/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
************************************************************************************************************************************************************************************************************

``TASK_QUEUE_ENABLE`` configures the optimization level of the task_queue operator dispatch queue, defaulting to Level 1. This configuration can reduce host dispatch time and can be used to alleviate the issue of excessive overall free time caused by dispatch.

.. image :: https://github.com/wucong25/verl-data/blob/main/en/figures/perf_tuning_task_queue.png
    :width: 500px

Level 0: Dispatch pipeline optimization is not enabled.

Level 1: Splits operator dispatch tasks into two stages. Some tasks (mainly aclnn operator calls) are placed on the newly added secondary pipeline. The primary and secondary pipelines pass tasks to each other through the operator queue and run in parallel. Through partial masking, the overall dispatch time is reduced and end-to-end performance is improved.

Level 2: Further balances the task load between the primary and secondary pipelines based on Level 1, mainly by migrating workspace-related tasks to the secondary pipeline. The masking effect is better and the performance gain is greater. This configuration takes effect only in binary scenarios; the recommended value is Level 2.

`Communication Algorithm Orchestration Expansion <https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
************************************************************************************************************************************************************************************************************
Use the environment variable ``HCCL_OP_EXPANSION_MODE=AIV`` to configure the orchestration expansion location of the communication algorithm. Supported values are as follows: 

- **AI_CPU:** The orchestration expansion location of the communication algorithm is the AI CPU on the Device side. The Device side automatically selects the corresponding scheduler based on the hardware model.

- **AIV:** The orchestration expansion location of the communication algorithm is the Vector Core on the Device side, and execution is also on the Vector Core.

- **HOST:** The orchestration expansion location of the communication algorithm is the Host-side CPU. The Device side automatically selects the corresponding scheduler based on the hardware model.

- **HOST_TS:** The orchestration expansion location of the communication algorithm is the Host-side CPU. The Host dispatches tasks to the Device's Task Scheduler, and the Device's Task Scheduler performs task scheduling and execution.

Inference Phase Tuning
--------------------------

Chunked Prefill in V1
***************************

The current version of vLLM has VLLM V1 enabled by default. Use the following configuration to enable Chunked Prefill:

.. code-block:: sh

    actor_rollout_ref.rollout.enable_chunked_prefill=True

For the principle, refer to the `vLLM official documentation <https://docs.vllm.ai/en/v0.4.2/models/performance.html>`_.

Graph Mode
***************************

Similar to CUDA, the NPU enables **ACL Graph** through the following configuration:

.. code-block:: sh

    actor_rollout_ref.rollout.enforce_eager=False

.. note::
    ACL Graph conflicts in principle with ``taskqueue Level 2``; **the two cannot be enabled at the same time**.

Training Phase Tuning
--------------------------

FSDP
**********************************

.. csv-table::
   :header: "FSDP", "Description"
   :widths: 30, 60

   "/","Shard optimizer only (Zero-1)"
   SHARD_GRAD_OP,Shard gradients and optimizer (Zero-2)
   "HYBRID_SHARD","Shard weights, gradients, and optimizer (Zero-3)"
   "2D device_mesh+HYBRID_SHARD","Also known as HSDP (FSDP+DDP). For example, device_mesh=[2,8]: every 8 ranks form one FSDP group, with FSDP sharding inside the group. There are two groups that perform DDP between them, synchronizing gradients through allreduce."
   "2D device_mesh+HYBRID_SHARD_ZERO2","The Zero-2 version of HSDP"
   NO_SHARD,DDP

FSDP does not support Zero-1. VeRL determines the device mesh value based on the number of cards and ``actor_rollout_ref.actor.fsdp_config.fsdp_size``, and uses Zero-3 for sharding by default. If the model is small (recommended to be smaller than 7B), you can set ``actor_rollout_ref.actor.fsdp_config.reshard_after_forward`` to ``True`` to use Zero-2 on FSDP/FSDP2 to optimize performance.

Megatron
**********************************

When the model is large, using Megatron as the training backend allows for more flexible performance tuning.

When DP parallelism cannot fit the model in memory, enable TP first to shard the model weights. If the model is still too large, enable PP for further sharding. If the sequence is too long, causing the activations to be too large, you can enable CP and SP for optimization. In MoE models, you can additionally enable EP to control expert sharding. If the experts are too small, to avoid sharding the weights too finely, you can enable ETP to avoid TP sharding in the MoE part and instead distribute multiple complete experts across DP and TP.

TP, PP, EP, and ETP are used in the same way as Megatron. The way to enable CP and SP on the NPU is as follows: 

- SP: ``Sequence Parallel`` further improves computation efficiency on top of Tensor Parallel. It is a parallel computation method that shards the input data along the sequence dimension. On the NPU, SP is invoked through MindSpeed:
  ::

      actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True

- CP: ``Context Parallel`` is a method for parallel processing of neural network activations across multiple GPUs/NPUs. It is implemented by partitioning the input tensor along the sequence dimension. On the NPU, CP is invoked through MindSpeed (the two parameters must be added together):
  ::

      actor_rollout_ref.actor.megatron.context_parallel_size
      actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
**********************************

When facing larger-sized models, it is usually necessary to shard the optimizer to each card within a DP domain to save memory. To enable the distributed optimizer on the NPU under the Megatron backend:

::

    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
