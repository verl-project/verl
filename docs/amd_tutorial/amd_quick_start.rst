Getting started with AMD ROCm
=========================================

Last updated: 05/17/2026.

Author: `Mingjie Lu <https://github.com/mingjielu>`_, `Xiaohong Kou <https://github.com/xiaohong42>`_, `Fuwei Yang <https://github.com/amd-fuweiy>`_

Overview
--------

This document is a quick-start tutorial for running VeRL on AMD ROCm.
It provides a production-style bring-up flow for container startup, environment
verification, and training examples.

Current software and hardware scope:

- Runtime modes: fully supports **Fully Async** and **Colocate**.
- Inference engine: **vLLM** validated; **SGLang** support is ongoing.
- Trainer backends: **FSDP**, **FSDP2** and **Megatron**.
- GPU targets:

  - MI300X / MI325X (``gfx942``)
  - MI355X (``gfx950``)

Software Baseline
-----------------

Use the following prebuilt image for tutorial and validation:

- ``amdagi/verl-dev:rocm7.0.2_56_te2.10_vllm0.20_py312``

The Docker build recipe remains unchanged:

- `docker/rocm/Dockerfile.rocm <https://github.com/verl-project/verl/blob/main/docker/rocm/Dockerfile.rocm>`_

Host Prerequisites
------------------

Before launching the container, ensure:

1. AMD ROCm 7.0.2 host driver stack is installed and healthy.
2. Docker has access to ``/dev/kfd`` and ``/dev/dri``.
3. Dataset and model storage paths are ready.

Launch Container
----------------

.. code-block:: bash

    NAME=verl_release
    DOCKER=amdagi/verl-dev:rocm7.0.2_56_te2.10_vllm0.20_py312

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
     - Status
     - Notes
   * - Runtime mode
     - Fully supported
     - Fully Async and Colocate are production-ready
   * - Inference engine
     - vLLM validated
     - SGLang integration is ongoing
   * - Trainer backend
     - Fully supported
     - FSDP, Megatron
   * - Hardware
     - Fully supported
     - MI300X / MI325X (gfx942), MI355X (gfx950)

Example Workflow
----------------

1) Colocate mode + FSDP (GRPO, Qwen3-8B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Qwen3-8B FSDP training, enable both parameter and optimizer offload to avoid OOM.

.. code-block:: bash

    # Configure these in your launch script or Hydra overrides:
    # actor_rollout_ref.actor.fsdp_config.param_offload=True
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

2) Colocate mode + Megatron (GRPO, Qwen3.5-35B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    bash examples/grpo_trainer/run_qwen3_5-35b-megatron.sh

3) Fully Async mode
~~~~~~~~~~~~~~~~~~~

``RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES`` and
``RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`` are no longer required in this release.

.. code-block:: bash

    # For qwen2.5-math-7b, update max_position_embeddings to 32768 in config.json after model download.
    bash verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh
