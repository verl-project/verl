Resolving Out-of-Memory (OOM) Errors
=====================================

Last updated: 12/30/2025.

This guide provides comprehensive solutions for resolving Out-of-Memory (OOM) errors in verl. OOM errors can occur on GPU or CPU during different phases of training (rollout, forward pass, backward pass, optimizer step). Each section addresses specific causes and solutions.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Diagnosis
---------------

Before diving into solutions, identify where OOM occurs:

1. **During rollout/inference**: Adjust ``gpu_memory_utilization`` and inference engine settings
2. **During forward pass**: Reduce ``micro_batch_size_per_gpu`` or enable gradient checkpointing
3. **During backward pass**: Enable gradient checkpointing and activation offloading
4. **During optimizer step**: Enable optimizer offloading
5. **CPU OOM**: Use memory-efficient allocators (tcmalloc/jemalloc)

GPU Memory Optimization
-----------------------

Rollout/Inference Phase
~~~~~~~~~~~~~~~~~~~~~~~

The inference engine (vLLM or SGLang) uses ``gpu_memory_utilization`` to control GPU memory allocation.

**Configuration:**

.. code-block:: yaml

   actor_rollout_ref:
     rollout:
       gpu_memory_utilization: 0.5  # Start conservative, increase gradually

**Key points:**

- For vLLM v0.7.0+: This controls the fraction of **total** GPU memory used
- For SGLang: This controls the fraction of **free** GPU memory for static allocations (weights + KV cache)
- When parameter/optimizer offload is enabled, you can push this to 0.8-0.9
- Without offload, keep between 0.5-0.7 to leave room for training

**CUDA Graphs Memory:**

CUDA graphs improve performance but consume additional memory that cannot be offloaded:

.. code-block:: yaml

   actor_rollout_ref:
     rollout:
       # Disable CUDA graphs to save memory
       enforce_eager: True

       # Or limit CUDA graph capture sizes
       cudagraph_capture_sizes: [1, 2, 4, 8, 16, 32]

Training Phase - Batch Size Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduce batch sizes to lower peak memory usage:

.. code-block:: yaml

   actor_rollout_ref:
     actor:
       # Reduce micro batch size
       ppo_micro_batch_size_per_gpu: 1  # Start small, increase if memory allows

       # Or use dynamic batching with token limits
       use_dynamic_bsz: True
       ppo_max_token_len_per_gpu: 8192  # Reduce this value if OOM

   critic:
     ppo_micro_batch_size_per_gpu: 2  # Critic can often use larger batches
     ppo_max_token_len_per_gpu: 16384

**Tips:**

- Forward-only operations (log_prob computation) can use larger batch sizes than backward operations
- Critic/Reward models can typically use 2x the batch size of Actor (smaller final layer)
- Set ``ppo_max_token_len_per_gpu`` to at least 2x ``(max_prompt_length + max_response_length)``

Gradient Checkpointing
~~~~~~~~~~~~~~~~~~~~~~

Gradient checkpointing trades compute for memory by recomputing activations during backward pass:

.. code-block:: yaml

   # For HuggingFace/FSDP backend
   actor_rollout_ref:
     model:
       enable_gradient_checkpointing: True

   critic:
     model:
       enable_gradient_checkpointing: True

   # For Megatron backend
   actor_rollout_ref:
     actor:
       megatron:
         override_transformer_config:
           recompute_method: uniform
           recompute_granularity: full
           recompute_num_layers: 1

Activation Offloading
~~~~~~~~~~~~~~~~~~~~~

Offload activations to CPU memory during forward pass, reload during backward:

.. code-block:: yaml

   actor_rollout_ref:
     model:
       enable_activation_offload: True
       enable_gradient_checkpointing: True  # Usually used together

   critic:
     model:
       enable_activation_offload: True
       enable_gradient_checkpointing: True

.. note::
   Activation offloading is currently available for the FSDP backend only.

Parameter, Gradient, and Optimizer Offloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large models, offload model states to CPU:

**Megatron Backend:**

.. code-block:: yaml

   actor_rollout_ref:
     actor:
       megatron:
         param_offload: True      # Offload parameters to CPU
         grad_offload: True       # Offload gradients to CPU
         optimizer_offload: True  # Offload optimizer states to CPU

     ref:
       megatron:
         param_offload: True      # Reference model only needs param offload

**FSDP Backend:**

.. code-block:: yaml

   actor_rollout_ref:
     actor:
       fsdp_config:
         cpu_offload: True
         offload_params: True

**CPU Optimizer (for very large models like DeepSeek):**

.. code-block:: yaml

   actor_rollout_ref:
     actor:
       optim:
         override_optimizer_config:
           optimizer_offload_fraction: 1.0
           overlap_cpu_optimizer_d2h_h2d: True
           use_precision_aware_optimizer: True
           optimizer_cpu_offload: True

Memory-Efficient Entropy Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Logits tensors with large vocabularies can consume significant memory:

.. code-block:: yaml

   actor_rollout_ref:
     ref:
       # Use chunked computation for entropy
       entropy_from_logits_with_chunking: True

     actor:
       # Enable entropy recomputation during training
       entropy_checkpointing: True

Liger Kernel Integration
~~~~~~~~~~~~~~~~~~~~~~~~

Liger Kernel provides fused, memory-efficient kernels for training:

.. code-block:: bash

   pip install liger-kernel

.. code-block:: yaml

   # For SFT
   model:
     use_liger: True

   # For RLHF
   actor_rollout_ref:
     model:
       use_liger: True

Benefits:

- Fused cross-entropy loss reduces memory for large vocabulary models
- Optimized RMSNorm, RoPE, and other operations
- Can significantly reduce peak memory usage

Ulysses Sequence Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For long context training, distribute sequence across GPUs:

.. code-block:: yaml

   actor_rollout_ref:
     actor:
       ulysses_sequence_parallel_size: 2  # Must divide number of GPUs

   critic:
     ulysses_sequence_parallel_size: 2

.. note::
   When training long sequences (>32k tokens), you may need to decrease
   ``micro_batch_size_per_gpu`` and ``max_token_len_per_gpu`` even with sequence parallelism.

FSDP2 Migration
~~~~~~~~~~~~~~~

FSDP2 offers ~7% lower GPU memory usage compared to FSDP1:

.. code-block:: yaml

   actor_rollout_ref:
     actor:
       strategy: fsdp2

Requirements: PyTorch 2.1+

CPU Memory Optimization
-----------------------

Using Memory-Efficient Allocators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default glibc malloc can cause memory fragmentation leading to CPU OOM. Use tcmalloc or jemalloc:

**Using tcmalloc:**

.. code-block:: bash

   # Install tcmalloc
   sudo apt-get install google-perftools libgoogle-perftools-dev

   # Set before running training
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

**Using jemalloc:**

.. code-block:: bash

   # Install jemalloc
   sudo apt-get install libjemalloc-dev

   # Set before running training
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

**For ARM/NPU platforms:**

.. code-block:: bash

   export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"

Reducing CPU Memory Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

- Use ``cpu_offload`` judiciously - while it saves GPU memory, it increases CPU memory usage
- When using multiple workers, be mindful of total CPU memory across all processes
- Consider reducing the number of data loading workers if CPU OOM occurs during data loading

Recommended Configuration Templates
-----------------------------------

Memory-Constrained Single GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   actor_rollout_ref:
     rollout:
       gpu_memory_utilization: 0.5
       enforce_eager: True

     model:
       enable_gradient_checkpointing: True
       enable_activation_offload: True
       use_liger: True

     actor:
       ppo_micro_batch_size_per_gpu: 1
       use_dynamic_bsz: True
       ppo_max_token_len_per_gpu: 4096

Multi-GPU with Large Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   actor_rollout_ref:
     rollout:
       gpu_memory_utilization: 0.7
       tensor_model_parallel_size: 4

     model:
       enable_gradient_checkpointing: True

     actor:
       megatron:
         param_offload: True
         optimizer_offload: True
       use_dynamic_bsz: True
       ppo_max_token_len_per_gpu: 16384

   critic:
     ppo_max_token_len_per_gpu: 32768

Very Large Models (100B+)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   actor_rollout_ref:
     rollout:
       gpu_memory_utilization: 0.85
       tensor_model_parallel_size: 8

     model:
       enable_gradient_checkpointing: True
       enable_activation_offload: True

     actor:
       megatron:
         param_offload: True
         grad_offload: True
         optimizer_offload: True
       optim:
         override_optimizer_config:
           optimizer_offload_fraction: 1.0
           overlap_cpu_optimizer_d2h_h2d: True

Debugging Memory Issues
-----------------------

Enable Memory Profiling
~~~~~~~~~~~~~~~~~~~~~~~

Use verl's built-in profiler to identify memory bottlenecks:

.. code-block:: yaml

   global_profiler:
     enabled: True
     global_tool_config:
       torch_memory:
         trace_alloc_max_entries: 100000
         stack_depth: 32

Monitor GPU Memory
~~~~~~~~~~~~~~~~~~

Add memory logging to your training:

.. code-block:: python

   from verl.utils.device import log_gpu_memory_usage
   import logging

   logger = logging.getLogger(__name__)
   log_gpu_memory_usage("Before forward pass", logger=logger)

Common Error Messages
~~~~~~~~~~~~~~~~~~~~~

- ``CUDA out of memory``: GPU memory exhausted - reduce batch sizes or enable offloading
- ``RuntimeError: CUDA error: an illegal memory access``: Often caused by vLLM issues - check vLLM version compatibility
- ``Killed`` or ``MemoryError``: CPU OOM - use tcmalloc/jemalloc and check offload settings

Summary Checklist
-----------------

When encountering OOM, try these solutions in order:

1. **Reduce batch sizes**: Lower ``ppo_micro_batch_size_per_gpu`` or ``ppo_max_token_len_per_gpu``
2. **Enable gradient checkpointing**: Set ``enable_gradient_checkpointing: True``
3. **Adjust inference memory**: Lower ``gpu_memory_utilization`` and consider ``enforce_eager: True``
4. **Enable activation offloading**: Set ``enable_activation_offload: True``
5. **Enable parameter/optimizer offloading**: For Megatron, set offload flags to True
6. **Use Liger Kernel**: Set ``use_liger: True`` for fused memory-efficient kernels
7. **For CPU OOM**: Use tcmalloc or jemalloc via ``LD_PRELOAD``
8. **Migrate to FSDP2**: Set ``strategy: fsdp2`` for ~7% memory savings
9. **Enable sequence parallelism**: For long contexts, increase ``ulysses_sequence_parallel_size``

See Also
--------

- :doc:`../perf/perf_tuning` - Performance tuning guide with memory optimization tips
- :doc:`../perf/best_practices` - Best practices including memory-related configurations
- `vLLM Optimization Guide <https://docs.vllm.ai/en/latest/performance/optimization.html>`_ - vLLM-specific tuning
