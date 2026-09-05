Ascend Performance Tuning Guide
====================================

Last updated:  01/29/2026.

Author:  `Xiaobo Hu <https://github.com/tardis-key>`_, `Haozhe Li <https://github.com/ZLiao097>`_

The performance tuning methods described in `Perf Tuning <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ are also applicable to Ascend devices. This guide highlights tuning methods specific to Ascend, including fused operator optimization, specific hardware configurations, Ascend affinity features, and so on.

Fusion operators
--------------------------

Common Fused Operators List
**********************************

The optimization principle of fused operators is to use mathematically equivalent substitution to fuse the computation of multiple operators into a single operator. This reduces redundant computation and the number of dispatches, thereby improving performance. Several typical NPU fused operators are listed below. Currently, they have all been replaced in `npu_patch.py` for the Qwen2 and Qwen3 series models.

For all fused operators used in the current verl, refer to `npu_patch.py <https://github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_

Matrix Computation-Communication operator fusion (MC2) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MC2 is a general term for a series of computation-communication fusion operators in CANN. These operators fuse originally sequential communication and computation operations, optimizing performance through internal splitting and pipeline parallel execution.

In vllm-ascend, you can specify environment variables:

.. code-block:: sh

    export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1

Enable ``torch_npu.npu_mm_all_reduce_base`` in the ``RowParallelLinear`` of the forward computation to merge the separate ``matmul`` and ``allreduce`` into a fused operator.

`RotaryMul&RotaryMulGrad <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu API:  ``torch_npu.npu_rotary_mul(x, r1, r2)``

Parameter description:

- x: q, k. The input shape must be 4-dimensional, typically ``[B, N, S, D]``, ``[B, S, N, D]``, or ``[S, B, N, D]``.

- r1: cos value. The shape requires the input to be 4-dimensional, typically ``[1, 1, S, D]``, ``[1, S, 1, D]``, or ``[S, 1, 1, D]``.

- r2: The sine value. The input shape must be 4-dimensional, which is generally ``[1, 1, S, D]``, ``[1, S, 1, D]``, or ``[S, 1, 1, D]``.

`RmsNorm&RmsNormGrad <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/torch_npu/(beta)torch_npu-npu_rms_norm.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface:  ``torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor)`` 
Parameter description:

- self: Tensor type, the shape supports 1 to 8 dimensions.

- gamma: A Tensor type, usually a weight. The shape must match the last few dimensions of self.

- epsilon: Float data type, used to prevent division by zero errors.

Output description:

- The first output is a Tensor, which is the final output y of the calculation formula.

- The second output is a Tensor, the intermediate result rstd of rms_norm, used for backward computation.

`Swiglu <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/torch_npu/(beta)torch_npu-npu_swiglu.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu API:  ``torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)``

Parameter description:

- self: Tensor type. The shape supports 1 to 8 dimensions.

- dim: Int type, defaults to -1.

Output description:

- The output is a Tensor, which is the final output y of the calculation formula.

`GroupMatMul <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

For detailed usage instructions, see the title document link.

Using fused operators with the FSDP backend
********************************************

In the ``verl/models/transformers/npu_patch.py`` file, available fused operators have been replaced using patches. You can use them by default without performing any other operations.

Using fused operators in the Megatron backend
**************************************************

Megatron fused operators are integrated in MindSpeed. You need to add specific parameters to enable them:

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

General Ascend Configuration
----------------------------

`Operator dispatch <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/ENV/docs/en/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
************************************************************************************************************************************************************************************************************

You can configure the task_queue operator dispatch queue optimization level using ``TASK_QUEUE_ENABLE``. The default is Level 1 optimization. This configuration can reduce host dispatch time and can be used to alleviate the issue of excessive overall free time caused by dispatch.

.. image :: https://github.com/verl-project/verl-data/blob/main/images/ascend/perf_tuning_task_queue.png
    :width: 500px

Level 0: Does not enable dispatch pipeline optimization.

Level 1 splits the operator dispatch task into two stages. The system places a part of the task (mainly the aclnn operator calls) on the newly added second-level pipeline. The first-level and second-level pipelines pass tasks through an operator queue and run in parallel. This partial overlap reduces the overall dispatch time and improves end-to-end performance.

Level 2 : Based on the Level 1 optimization, this level further balances the task load between the first-level and second-level pipelines. It mainly migrates workspace-related tasks to the second-level pipeline, achieving better latency hiding and greater performance gains. This configuration takes effect only in binary scenarios. The recommended configuration value is Level 2 optimization.

`Communication Algorithm Orchestration Expansion <https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
************************************************************************************************************************************************************************************************************
Use the environment variable ``HCCL_OP_EXPANSION_MODE=AIV`` to configure the orchestration expansion location of the communication algorithm. The following values are supported:

- **AI_CPU:** Indicates that the communication algorithm is orchestrated and expanded on the AI CPU on the Device side. The Device side automatically selects the corresponding scheduler based on the hardware model.

- **AIV:** Indicates that the communication algorithm is orchestrated and expanded on the Vector Core on the device side, and execution also takes place on the Vector Core.

- **HOST:** Indicates that the communication algorithm is orchestrated and expanded on the host-side CPU. The device side automatically selects the corresponding scheduler based on the hardware model.

- **HOST_TS:** Indicates that the orchestration and expansion of the communication algorithm occur on the host-side CPU. The Host sends tasks to the Device's Task Scheduler, which then schedules and executes the tasks.

Inference Phase Tuning
--------------------------

Chunked Prefill in V1
***************************

The current version of VLLM enables VLLM V1 by default. Use the following configuration to enable Chunked Prefill:

.. code-block:: sh

    actor_rollout_ref.rollout.enable_chunked_prefill=True

For the underlying principles, refer to the `official VLLM documentation <https://docs.vllm.ai/en/v0.4.2/models/performance.html>`_.

Graph Mode
***************************

Similar to CUDA, the NPU enables **ACL Graph** using the following configuration:

.. code-block:: sh

    actor_rollout_ref.rollout.enforce_eager=False

.. note::
    ACL Graph and ``taskqueue Level 2`` conflict in principle, and **they cannot be enabled simultaneously**.

Training phase tuning
--------------------------

FSDP
**********************************

.. csv-table::
   :header: "FSDP", "Description"
   :widths: 30, 60

   "/","Shards only the optimizer (Zero-1)"
   SHARD_GRAD_OP,Shards gradients and optimizer (Zero-2)
   "HYBRID_SHARD","Shards weights, gradients, and optimizer (Zero-3)"
   "2D device_mesh+HYBRID_SHARD","Also known as HSDP (FSDP+DDP). For example, with device_mesh=[2,8], every 8 ranks form an FSDP group. FSDP sharding is performed within each group. There are two groups in total, and DDP is performed between the two groups. Gradients are synchronized through allreduce."
   "2D device_mesh+HYBRID_SHARD_ZERO2","The Zero-2 version of HSDP"
   NO_SHARD,DDP

FSDP does not support Zero-1. VeRL determines the device mesh value based on the number of devices and ``actor_rollout_ref.actor.fsdp_config.fsdp_size``, and uses Zero-3 for sharding by default. If the model is small (less than 7B is recommended), set the parameter ``actor_rollout_ref.actor.fsdp_config.reshard_after_forward`` to ``True`` to use Zero-2 on FSDP/FSDP2 to optimize performance.

Megatron
**********************************

When the model is large, using Megatron as the training backend enables more flexible performance tuning.

When the device memory of DP parallelism cannot accommodate the model, enable TP first to split the model weights. If the model is still too large, enable PP to further split it. If the sequence is too long and the activations become too large, you can enable CP and SP for optimization. In MoE models, you can additionally enable EP to control the splitting of experts. If the experts are too small, to avoid splitting the weights too finely, you can enable ETP to avoid TP splitting in the MoE part. This distributes multiple complete experts across DP and TP.

TP, PP, EP, ETP, and Megatron are used in the same way. To enable CP and SP on the NPU:

- SP: ``Sequence Parallel`` builds upon Tensor Parallel to further improve computing efficiency. It is a parallel computing method that splits the sequence dimension of the input data. On the NPU, you can invoke SP using MindSpeed:
  ::

      actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True

- CP: ``Context Parallel`` processes neural network activations in parallel across multiple GPUs/NPUs by partitioning the input tensor along the sequence dimension. On the NPU, invoke CP using MindSpeed (both parameters must be added simultaneously):
  ::

      actor_rollout_ref.actor.megatron.context_parallel_size
      actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
**********************************

When handling larger models, you typically need to shard the optimizer to each device within a DP domain to save device memory. To enable the distributed optimizer on the NPU using the Megatron backend:

::

    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
