verl performance tuning for AMD (ROCm Kernel)
=====================================================

Last updated: 11/13/2025.

Author: `Yang Wang <https://github.com/YangWang92/>`_, `Songlin Jiang <https://github.com/HollowMan6/>`_

Use vLLM Sleep Mode for AMD MI3xx series GPUs
--------------------------------------------------------------

By default, verl requires vLLM to enable sleep mode, which allows vLLM to offload GPU memory to CPU memory after rollout. This feature has been merged into the main branch of vLLM for version later than 0.11.0.

For now, you can use the vLLM main branch and build it from the source code, or you can directly install vLLM from the pre-built ROCm wheels for vLLM version later than 0.11.0 when it's available.

1. Clone the vLLM repository and build it with the following commands:

.. code-block:: bash

    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git reset --hard 4ca5cd5740c0cd7788cdfa8b7ec6a27335607a48 # You can also use a later commit as you wish
    python -m pip install -r requirements/rocm.txt
    VLLM_TARGET_DEVICE=rocm ROCM_PATH=/opt/rocm/ python3 setup.py develop

2. Additionally, we recommend you to use the ROCm version later than or equal to ROCm 7.0.

After the upgrade, you can verify whether sleep mode is working by trying out `these scripts <https://github.com/EmbeddedLLM/inference-experiment/tree/main/sleep_mode>`_.

If sleep mode is working, you should see the memory usage reduce after sleep.

After applying the vLLM patch and completing the installation, you can enable sleep mode in verl to reduce memory overhead. This allows verl to offload unused GPU memory during rollout, significantly lowering the memory footprint during long-context training or multi-node reinforcement learning.


Enable CUDA Graph and Bypass ROCm-related issues
--------------------------------------------------------------

Due to potential issues with CUDA graph capture in ROCm, we've found that vLLM's CUDA graph feature cannot be enabled on multiple nodes in verl on AMD platforms with vLLM V1 mode. This leads to significantly slower rollout performance.

Our investigation shows that ROCm may trigger an unexpected crash when attempting to capture large batches with CUDA graph. One workaround is to set ``actor_rollout_ref.rollout.cudagraph_capture_sizes`` to values such as ``[1, 2, 4, 8, 16, 32, 64]`` (change depending on your GPU memory size).

Then, you can choose to enable CUDA graph by setting ``actor_rollout_ref.rollout.enforce_eager`` to ``False`` in your verl configuration file.
