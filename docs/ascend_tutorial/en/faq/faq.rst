NPU Frequently Asked Questions
===============================

Last updated: 05/13/2026.

This document summarizes common issues and their solutions that you may encounter when running VERL training and inference on an NPU.

Environment Configuration Issues
--------------------------------

### Q1: What should you do if the NPU device is not visible?

**Symptom**: torch_npu.npu.is_available() returns False

**Solution**:

.. code-block:: bash

   # Check device visibility
   echo $ASCEND_RT_VISIBLE_DEVICES

   # Set the visible devices and disable automatic setting by Ray
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

   # Check the driver status
   npu-smi info

Debugging and diagnostics
-------------------------

### Q1: How do I enable NPU performance profiling?

Use the VERL built-in profiler:

.. code-block:: shell

   actor_rollout_ref.actor.profiler.tool_config.npu.discrete=true \
   actor_rollout_ref.actor.profiler.tool_config.npu.contents=npu,cpu \
   actor_rollout_ref.actor.profiler.tool_config.npu.level=1 \
   actor_rollout_ref.actor.profiler.tool_config.npu.analysis=true

### Q2: How to troubleshoot NPU training failures?

**Troubleshooting steps**:

1. Check the environment variable configuration
2. Verify device visibility
3. Check CANN version compatibility
4. View specific error messages in the logs
5. Reproduce the issue using a minimal sample

**Enable verbose logging**:

.. code-block:: bash

   # VERL framework logs
   export VERL_LOGGING_LEVEL=DEBUG

   # Ascend NPU log (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
   export ASCEND_GLOBAL_LOG_LEVEL=0
   export ASCEND_SLOG_PRINT_TO_STDOUT=1

   # HCCL communication logs
   export HCCL_DEBUG=INFO

Common error messages
----------------------

### Q1： "torch_npu detected, but NPU device is not available or visible"

**Cause**: The NPU driver is not installed correctly or the device is not visible

**Solution**: Check the driver installation status and the ASCEND_RT_VISIBLE_DEVICES setting.

### Q2： "KeyError: decoder.layers.0.self_attention.q_layernorm.weight"

**Cause**: The MindSpeed version is too low

**Solution**: Switch MindSpeed to 2.3.0_core_r0.12.1

### Q3： "AssertionError: Weight ... is too large to fit in the bucket"

**Symptom**: The following error occurs during weight synchronization in distributed training:

.. code-block:: text

   AssertionError: Weight model.embed_tokens.weight(torch.Size([151936, 4096]), torch.float32) is too large to fit in the bucket.
   Please increase rollout.update_weights_bucket_megabytes(2048 MB).

**Cause**: The size of a model weight tensor exceeds the default capacity of the weight transmission bucket (2048 MB). In the verl framework, model weights are transmitted in chunks through a bucket (buffer). When a single weight tensor exceeds the bucket size, the assertion check fails.

**Weight size calculation method**:

The memory usage of the weight tensor (bytes) = the product of all dimension sizes × the number of bytes per element

The number of bytes for each data type is as follows:

- ``torch.float32`` → 4 bytes
- ``torch.float16`` / ``torch.bfloat16`` → 2 bytes
- ``torch.int8`` → 1 byte

Take ``model.embed_tokens.weight`` in this example as an example:

.. code-block:: text

   Tensor shape: torch.Size([151936, 4096])
   Data type: torch.float32 (4 bytes)
   Weight size = 151936 × 4096 × 4 = 2,483,027,968 bytes ≈ 2369 MB

   The default bucket size = 2048 MB < 2369 MB → triggers an assertion failure

**Solution**: Add the ``update_weights_bucket_megabytes`` parameter when starting training so that the bucket capacity is greater than the memory usage of the largest weight tensor:

.. code-block:: bash

   actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096

**Recommended parameter values**:

1. **Calculate the memory usage of the largest weight tensor in the model**: Iterate through all model parameters, find the one with the largest ``nbytes``, and convert it to MB (divide by 1024²).

2. **Round up to the nearest power of 2**: To facilitate memory allocation and management, we recommend rounding up the calculation result to the nearest power of 2 (such as 2048, 4096, 8192, and so on). For example, if the maximum weight is 2369 MB, use 4096 MB.

3. **Reserve an appropriate margin**: To account for memory alignment and runtime overhead, set the bucket size to at least 1.2 to 1.5 times the maximum weight size, and then round up to the nearest power of 2.

4. **Pay attention to memory limits**: The bucket size directly affects the memory usage of worker nodes. Setting it too large causes OOM. Choose a value as small as possible while meeting the weight transmission requirements.

**Recommended values for common models**:

.. list-table::
   :header-rows: 1

   * - Model size
     - Typical maximum weight shape
     - Recommended bucket size
   * - 7B (Qwen2 and so on)
     - [151936, 4096] float32
     - 4096 MB
   * - 14B
     - [152064, 5120] float32
     - 4096 MB
   * - 72B
     - [152064, 8192] float32
     - 8192 MB

### Q4: Checkpoint loading fails in non-shared storage, and common.pt / .metadata / metadata.json cannot be found

**Symptom**: When using the verl + Megatron backend in a multi-node environment with **non-shared storage**, saving checkpoints works normally, but an error occurs during reloading, indicating that the following file cannot be found:

.. code-block:: text

   FileNotFoundError: common.pt
   FileNotFoundError: .metadata
   FileNotFoundError: metadata.json

**Cause**: The current checkpoint mechanism does not fully support non-shared storage. Specifically, this manifests as:

- **Distributed training weights are saved on a per-node basis**. Each node saves only the weight shards it is responsible for, rather than saving all weights only on the primary node.
- However, metadata files such as ``common.pt``, ``.metadata``, and ``metadata.json`` **are saved only on the node that performs the save operation** (usually the node where rank 0 is located). Other nodes do not have these files locally.
- When you load a checkpoint, each node needs to read these metadata files to restore the model state. However, without shared storage, these files do not exist in the local paths of other nodes, causing a loading failure.

**Temporary solution**: Manually copy the metadata file from the saving node to all other nodes:

.. code-block:: bash

   # Assume the checkpoint is saved in the /path/to/ckpt/ directory on the rank 0 node
   # Copy the metadata file from the rank 0 node to all other nodes

   # Files to copy
   /path/to/ckpt/common.pt
   /path/to/ckpt/.metadata
   /path/to/ckpt/metadata.json

   # Example: Copy to other nodes using scp
   scp /path/to/ckpt/common.pt node1:/path/to/ckpt/
   scp /path/to/ckpt/.metadata node1:/path/to/ckpt/
   scp /path/to/ckpt/metadata.json node1:/path/to/ckpt/

   # Repeat the preceding operations for all nodes

**Precautions**:

- You must copy the metadata files again after each checkpoint save, because the save operation might update their contents.
- If you frequently save checkpoints during training (for example, automatic saving by steps), write a script to automatically trigger copying after saving to prevent omissions.
- For the long-term solution, wait for the framework to support loading checkpoints from non-shared storage, allowing metadata files to automatically synchronize to all nodes.

References
----------

- `Ascend Performance Tuning Guide <../dev_guide/performance/perf_tuning_on_ascend.rst>`_
- `Ascend Quick Start Guide <../get_start/quick_start.rst>`_
- `NPU-CI Addition Guide <../contribution_guide/ascend_ci_guide.rst>`_
- Ascend NPU documentation: https://www.hiascend.com/en/document
- CANN toolkit documentation: https://www.hiascend.com/eng/cann

Get more help
-------------

If the preceding FAQ does not resolve your issue:

1. View the complete error log.
2. Search for similar issues in GitHub Issues.
3. Provide detailed error information and the environment configuration.
4. Provide a minimal reproducible example.