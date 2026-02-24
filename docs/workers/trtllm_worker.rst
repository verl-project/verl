TensorRT-LLM Backend
====================

Last updated: 12/31/2025.

**Authored By TensorRT-LLM Team**

Introduction
------------
`TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`_ is a high-performance LLM inference engine with state-of-the-art optimizations for NVIDIA GPUs.
The verl integration of TensorRT-LLM is based on TensorRT-LLM's `Ray orchestrator <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/ray_orchestrator>`_. This integration is in its early stage, with more features and optimizations to come.

The TensorRT-LLM rollout engine primarily targets the colocated mode. Instead of relying purely on standard colocated mode, we adopted a mixed design combining aspects of the hybrid engine and colocated mode.

Installation
------------
We provide ``docker/Dockerfile.stable.trtllm`` for building a docker image with TensorRT-LLM pre-installed. The verl integration is supported from ``nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6``, and you can choose other TensorRT-LLM versions via ``TRTLLM_BASE_IMAGE`` from the `NGC Catalog <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release>`_.

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

Using TensorRT-LLM as the Rollout Engine for GRPO
-------------------------------------------------

We provide the following GRPO recipe scripts for you to test the performance and accuracy curve of TensorRT-LLM as the rollout engine:

.. code-block:: bash

    ## For FSDP training engine
    bash examples/grpo_trainer/run_qwen2-7b_math_trtllm.sh
    ## For Megatron-Core training engine
    bash examples/grpo_trainer/run_qwen2-7b_math_megatron_trtllm.sh

Using TensorRT-LLM as the Rollout Engine for DAPO
-------------------------------------------------

We provide a DAPO recipe script ``recipe/dapo/test_dapo_7b_math_trtllm.sh``.

.. code-block:: bash

    ## For FSDP training engine
    bash recipe/dapo/test_dapo_7b_math_trtllm.sh
    ## For Megatron-Core training engine
    TRAIN_ENGINE=megatron bash recipe/dapo/test_dapo_7b_math_trtllm.sh

