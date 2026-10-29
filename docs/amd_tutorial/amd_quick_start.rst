Getting started with AMD ROCm
=========================================

Last updated: 07/24/2026.

Author: `Mingjie Lu <https://github.com/mingjielu>`_, `Xiaohong Kou <https://github.com/xiaohong42>`_, `Fuwei Yang <https://github.com/amd-fuweiy>`_, `Zhaodong Bing <https://github.com/aaab8b>`_

Overview
--------

This document is a quick-start tutorial for running VeRL on AMD ROCm.
It provides a production-style bring-up flow for container startup, environment
verification, and training examples.

Current software and hardware scope:

- Runtime modes: fully supports **Fully Async** and **Colocate**.
- Inference engine: fully supports **vLLM** and **SGLang**.
- Trainer backends: **FSDP**, **FSDP2** and **Megatron**.
- GPU targets:

  - MI300X / MI325X (``gfx942``)
  - MI350X / MI355X (``gfx950``)

Software Baseline
-----------------

Use the following prebuilt image for tutorial and validation:

- ``amdagi/verl-dev:rocm7.14_torch2.12_release_0724``

Or build from source:

- `docker/rocm/Dockerfile.rocm <https://github.com/verl-project/verl/blob/main/docker/rocm/Dockerfile.rocm>`_

Host Prerequisites
------------------

Before launching the container, ensure:

1. AMD ROCm 7.14 host driver stack is installed and healthy.
2. Docker has access to ``/dev/kfd`` and ``/dev/dri``.
3. Dataset and model storage paths are ready.

Launch Container
----------------

.. code-block:: bash

    NAME=verl_release
    DOCKER=amdagi/verl-dev:rocm7.14_torch2.12_release_0724

    docker pull $DOCKER

    docker run -it --name $NAME --device /dev/kfd --device /dev/dri \
      --privileged --network=host \
      --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      --shm-size=2048g \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -w /workspace \
      $DOCKER \
      /bin/bash

Environment Check (Inside Container)
------------------------------------

.. code-block:: bash

    # ROCm and visible GPU targets
    rocminfo | grep -E "gfx942|gfx950" || true

    # PyTorch + ROCm sanity check
    python - <<'PY'
    import torch
    print("torch:", torch.__version__)
    print("rocm :", torch.version.hip)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_count:", torch.cuda.device_count())
        print("device_0:", torch.cuda.get_device_name(0))
    PY

Feature Support Matrix
----------------------

.. list-table:: Current support status
   :header-rows: 1

   * - Category
     - Notes
   * - Runtime mode
     - Fully Async, Colocate 
   * - Inference engine
     - vLLM, SGLang 
   * - Trainer backend
     - FSDP, FSDP2, Megatron
   * - Hardware
     - MI300X / MI325X (gfx942), MI350X/MI355X (gfx950)

Example Workflow
----------------

.. list-table:: 
   :header-rows: 1

   * - Runtime mode
     - Inference engine
     - Trainer backend
     - Example scripts
   * - Colocate
     - vLLM
     - FSDP
     - `bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh <../../examples/grpo_trainer/run_qwen3_8b_fsdp.sh>`_
   * - Colocate
     - vLLM
     - Megatron
     - `bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh <../../examples/grpo_trainer/run_qwen3_5_35b_megatron.sh>`_
   * - Fully Async
     - vLLM
     - FSDP2
     -  `bash verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh <../../verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh>`_
   * - Fully Async
     - vLLM
     - Megatron
     -  `bash verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh <../../verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh>`_
   * - Colocate
     - SGLang
     - FSDP
     - `bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh <../../examples/grpo_trainer/run_qwen3_8b_fsdp.sh>`_ (vllm->sglang)
   * - Colocate
     - SGLang
     - Megatron
     - `bash examples/grpo_trainer/run_qwen3_8b_megatron.sh <../../examples/grpo_trainer/run_qwen3_8b_megatron.sh>`_ (vllm->sglang)
   * - Fully Async
     - SGLang
     - FSDP2
     -  `bash verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh <../../verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh>`_ (vllm->sglang)
   * - Fully Async
     - SGLang
     - Megatron
     -  `bash verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh <../../verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh>`_ (vllm->sglang)
   
   


Known Issues
----------------
1. For qwen2.5-math-7b, update max_position_embeddings to 32768 in config.json after model download.
2. ``PYTORCH_ALLOC_CONF=expandable_segments:True`` is set by default in ``Dockerfile.rocm`` to prevent out-of-memory (OOM). However, this setting may conflict with ``vllm_custom_all_reduce``, so we set ``vllm.disable_custom_all_reduce=True`` in ``config.yaml`` by default which will be removed in the future once ROCm resolves the conflict.
3. ``attention_backend`` must be set to ``triton`` for SGLang.
4. Several ROCm-specific pull requests for vLLM and SGLang are still pending upstream integration, so we currently incorporate the required patches directly within the Dockerfile. In the future, we intend to transition to stable, officially released versions.

