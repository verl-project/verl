TensorRT-LLM Backend
====================

Last updated: 5/6/2026.

**Authored By NVIDIA TensorRT-LLM Team**

Introduction
------------
`TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`_ is a high-performance LLM inference engine with state-of-the-art optimizations for NVIDIA GPUs.
The verl integration of TensorRT-LLM is based on TensorRT-LLM's `Ray orchestrator <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/ray_orchestrator>`_, with more features and performance optimizations to come.

- For **synchronous training**, the TensorRT-LLM rollout adopts a mixed design combining aspects of the hybrid engine and colocated mode, instead of relying purely on standard colocated mode.
- For **asynchronous training**, the TensorRT-LLM rollout follows other rollout backends and uses standalone mode for trainer and rollout placement.

TensorRT-LLM rollout supports the following key features, primarily tested on Qwen3 dense and MoE variants:

- Synchronous training (GRPO, DAPO, etc.)
- Cross-node inference
- FP8 refit
- Asynchronous training (further optimizations planned)
- Preliminary support for VLM

You can track our roadmap and share feedback at the `TensorRT-LLM rollout roadmap <https://github.com/verl-project/verl/issues/5042>`_.


Installation
------------
We recommend using `docker/Dockerfile.stable.trtllm <https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.trtllm>`_ for building a docker image with TensorRT-LLM pre-installed. The verl integration is supported from ``nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6``, and you can choose other TensorRT-LLM versions via ``TRTLLM_BASE_IMAGE`` from the `NGC Catalog <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release>`_. The image is updated periodically to track TensorRT-LLM's weekly releases.

Alternatively, refer to the `TensorRT-LLM installation guide <https://nvidia.github.io/TensorRT-LLM/installation/index.html>`_ for compatible environments if you want to build your own.

Install verl with TensorRT-LLM:

.. code-block:: bash

    pip install --upgrade pip
    pip install -e ".[trtllm]"

.. note::

    Using the TensorRT-LLM rollout requires setting the following environment variables before launching the Ray cluster. These have been included in all the example scripts:

    .. code-block:: bash

        # Clean all SLURM/MPI/PMIx env to avoid PMIx mismatch error.
        for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
            unset "$v"
        done

Using TensorRT-LLM rollout for GRPO
------------------------------------

.. code-block:: bash

    ## For FSDP training engine
    INFER_BACKEND=trtllm bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
    ## For Megatron-Core training engine
    INFER_BACKEND=trtllm bash examples/grpo_trainer/run_qwen3_8b_megatron.sh

Using TensorRT-LLM rollout for DAPO with FP8
---------------------------------------------

.. code-block:: bash

    # For Megatron-Core training engine with FP8 rollout
    INFER_BACKEND=trtllm ROLLOUT_QUANTIZATION=fp8 bash examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh

Using TensorRT-LLM rollout in fully async with GRPO
----------------------------------------------------
.. code-block:: bash

    # Fully async policy with Megatron-Core training engine
    bash verl/experimental/fully_async_policy/shell/grpo_30b_a3b_base_math_megatron_4_4_mis_trtllm.sh
