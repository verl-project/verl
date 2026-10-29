NPU FAQ
=======

Last updated: 05/13/2026.

This document summarizes common issues and solutions encountered when running verl training and inference on NPU.

Environment Configuration Issues
--------------------------------

### Q1: What should I do if the NPU device is not visible?

**Symptom**: ``torch_npu.npu.is_available()`` returns ``False``

**Solution**:

.. code-block:: bash

   # Check device visibility
   echo $ASCEND_RT_VISIBLE_DEVICES

   # Set visible devices and disable automatic setting by ray
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

   # Check driver status
   npu-smi info

Debugging and Diagnostics
-------------------------

### Q1: How do I enable NPU profiling?

Use verl's built-in profiler:

.. code-block:: shell

   actor_rollout_ref.actor.profiler.tool_config.npu.discrete=true \
   actor_rollout_ref.actor.profiler.tool_config.npu.contents=npu,cpu \
   actor_rollout_ref.actor.profiler.tool_config.npu.level=1 \
   actor_rollout_ref.actor.profiler.tool_config.npu.analysis=true

### Q2: How do I troubleshoot NPU training failures?

**Troubleshooting steps**:

1. Check environment variable configuration
2. Verify device visibility
3. Check CANN version compatibility
4. Review the specific error messages in the logs
5. Reproduce the issue with a minimal example

**Enable verbose logging**:

.. code-block:: bash

   # verl framework logs
   export VERL_LOGGING_LEVEL=DEBUG

   # Ascend NPU logs (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
   export ASCEND_GLOBAL_LOG_LEVEL=0
   export ASCEND_SLOG_PRINT_TO_STDOUT=1

   # HCCL communication logs
   export HCCL_DEBUG=INFO

Common Error Messages
---------------------

### Q1: "torch_npu detected, but NPU device is not available or visible"

**Cause**: The NPU driver is not installed correctly, or the device is not visible.

**Solution**: Check the driver installation status and the ``ASCEND_RT_VISIBLE_DEVICES`` setting.

### Q2: "KeyError: decoder.layers.0.self_attention.q_layernorm.weight"

**Cause**: MindSpeed version is too old.

**Solution**: Switch MindSpeed to ``2.3.0_core_r0.12.1``.

### Q3: "AssertionError: Weight ... is too large to fit in the bucket"

**Symptom**: The following error occurs during weight synchronization in distributed training:

.. code-block:: text

   AssertionError: Weight model.embed_tokens.weight(torch.Size([151936, 4096]), torch.float32) is too large to fit in the bucket.
   Please increase rollout.update_weights_bucket_megabytes(2048 MB).

**Cause**: The size of a weight tensor exceeds the default capacity (2048 MB) of the weight-transfer bucket. In verl, model weights are transferred in chunks through buckets (buffers). When a single weight tensor is larger than the bucket size, the assertion check fails.

**How to compute the weight size**:

The memory footprint (in bytes) of a weight tensor = the product of all dimension sizes × the number of bytes per element.

The number of bytes for each data type is:

- ``torch.float32`` → 4 bytes
- ``torch.float16`` / ``torch.bfloat16`` → 2 bytes
- ``torch.int8`` → 1 byte

Taking ``model.embed_tokens.weight`` from this example:

.. code-block:: text

   Tensor shape: torch.Size([151936, 4096])
   Data type: torch.float32 (4 bytes)
   Weight size = 151936 × 4096 × 4 = 2,483,027,968 bytes ≈ 2369 MB

   Default bucket size = 2048 MB < 2369 MB → assertion failure triggered

**Solution**: Increase the ``update_weights_bucket_megabytes`` parameter when starting training so that the bucket capacity is larger than the memory footprint of the largest weight tensor:

.. code-block:: bash

   actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096

**Recommendations for choosing the parameter value**:

1. **Compute the memory footprint of the largest weight tensor in the model**: Iterate over all model parameters, find the one with the largest ``nbytes``, and convert it to MB (divide by 1024²).

2. **Round up to the next power of two**: For easier memory allocation and management, round the result up to the nearest power of two (for example, 2048, 4096, 8192). For example, if the largest weight is 2369 MB, choose 4096 MB.

3. **Leave an appropriate margin**: Considering memory alignment and runtime overhead, the bucket size should be at least 1.2–1.5× the size of the largest weight, then rounded up to the next power of two.

4. **Be aware of memory limits**: The bucket size directly affects the memory usage of worker nodes. Setting it too large will cause OOM. Choose the smallest value that still satisfies the weight-transfer requirement.

**Recommended values for common models**:

.. list-table::
   :header-rows: 1

   * - Model Size
     - Typical Largest Weight Shape
     - Recommended Bucket Size
   * - 7B (Qwen2, and so on)
     - [151936, 4096] float32
     - 4096 MB
   * - 14B
     - [152064, 5120] float32
     - 4096 MB
   * - 72B
     - [152064, 8192] float32
     - 8192 MB

### Q4: Checkpoint loading fails on non-shared storage with common.pt / .metadata / metadata.json not found

**Symptom**: When using verl + Megatron backend in a multi-node environment with **non-shared storage**, saving checkpoints works but reloading fails with errors indicating the following files cannot be found:

.. code-block:: text

   FileNotFoundError: common.pt
   FileNotFoundError: .metadata
   FileNotFoundError: metadata.json

**Cause**: The current checkpoint mechanism does not fully support non-shared storage. Specifically:

- **In distributed training, weights are saved per node**: each node only saves the weight shards it is responsible for, and not all weights are saved on the master node.
- However, metadata files such as ``common.pt``, ``.metadata``, and ``metadata.json`` are **only saved on the node that performs the save operation** (usually the rank 0 node); other nodes do not have these files locally.
- When loading a checkpoint, every node needs to read these metadata files to restore the model state, but under non-shared storage these files are absent from the local paths of the other nodes, causing the load to fail.

**Temporary workaround**: Manually copy the metadata files from the saving node to all other nodes:

.. code-block:: bash

   # Assume the checkpoint is saved under /path/to/ckpt/ on the rank 0 node
   # Copy the metadata files from the rank 0 node to all other nodes

   # Files to copy
   /path/to/ckpt/common.pt
   /path/to/ckpt/.metadata
   /path/to/ckpt/metadata.json

   # Example: copy to another node using scp
   scp /path/to/ckpt/common.pt node1:/path/to/ckpt/
   scp /path/to/ckpt/.metadata node1:/path/to/ckpt/
   scp /path/to/ckpt/metadata.json node1:/path/to/ckpt/

   # Repeat the above for all nodes

**Precautions**:

- After every checkpoint save, you need to re-copy the metadata files, because the save operation may update their contents.
- If checkpoints are saved frequently during training (for example, automatic step-based saving), it is recommended to write a script that automatically triggers the copy after each save to avoid omissions.
- The long-term solution is to wait for framework-level support for loading checkpoints on non-shared storage, so that metadata files can be automatically synchronized to all nodes.

References
----------

- `Ascend Performance Tuning Guide <../dev_guide/performance/perf_tuning_on_ascend.rst>`_
- `Ascend Quick Start Guide <../get_start/quick_start.rst>`_
- `NPU-CI Contribution Guide <../contribution_guide/ascend_ci_guide.rst>`_
- Ascend NPU documentation: https://www.hiascend.com/en/document
- CANN toolkit documentation: https://www.hiascend.com/eng/cann

Getting More Help
-----------------

If the above FAQ does not resolve your issue:

1. Review the complete error log
2. Search for similar issues in GitHub Issues
3. Provide detailed error information and environment configuration
4. Provide a minimal reproducible example
