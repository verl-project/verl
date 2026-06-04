Getting Started with Intel GPU
==============================

Last updated: 06/04/2026.

Author: `Kah Lun Teoh <https://github.com/kahlun>`_

Overview
--------

This document is a quick-start tutorial for running verl on Intel GPU
(Arc / Arc Pro).

It covers environment verification and training examples using FSDP + vLLM
rollout in colocated mode.

Current software and hardware scope:

- Runtime mode: **Colocate** (FSDP actor + vLLM rollout on the same device).
- Inference engine: **vLLM** validated.
- Trainer backend: **FSDP**, **FSDP2**.
- Algorithms: GRPO, PPO, SFT.
- Hardware targets:

  - Intel Arc Pro B60 (Battlemage, 24 GB)
  - Intel Arc Pro B70 (2× GPU, 32 GB each)

For the Docker image software stack (vLLM 0.17.1, compute-runtime 26.09, and
the corresponding torch/triton XPU versions resolved from vLLM
``requirements/xpu.txt``), build instructions, and environment variables
reference, see :doc:`intel_gpu_build_dockerfile_page`.

Environment Check (Inside Container)
--------------------------------------

After launching the container (see :doc:`intel_gpu_build_dockerfile_page`):

.. code-block:: bash

    # Confirm Intel GPU devices are visible
    python3 - <<'PY'
    import torch
    print("torch:", torch.__version__)
    print("xpu_available:", torch.xpu.is_available())
    print("device_count:", torch.xpu.device_count())
    for i in range(torch.xpu.device_count()):
        print(f"  device_{i}:", torch.xpu.get_device_name(i))
    PY

    # Confirm oneCCL runtime env and shared library are available.
    # The Docker image does not currently ship a separate Python binding module.
    python3 - <<'PY'
    import ctypes.util
    import os
    print("CCL_ROOT:", os.environ.get("CCL_ROOT"))
    print("libccl:", ctypes.util.find_library("ccl"))
    PY

Expected output (2× Arc Pro B60):

.. code-block:: text

    torch: 2.11.0+xpu
    xpu_available: True
    device_count: 2
      device_0: Intel(R) Arc(TM) Pro B60 Graphics
      device_1: Intel(R) Arc(TM) Pro B60 Graphics
    CCL_ROOT: /opt/intel/oneapi/ccl/2021.15
    libccl: libccl.so.1

  .. note::

     On the current image, oneCCL is validated through its runtime environment
     and shared library availability rather than a Python import. During GRPO/PPO
     launches you should also see oneCCL startup messages in the worker logs.

Feature Support Matrix
-----------------------

.. list-table::
   :header-rows: 1

   * - Category
     - Status
     - Notes
   * - Runtime mode
     - Colocate
     - FSDP/FSDP2 actor + vLLM rollout on same GPU(s)
   * - Inference engine
     - vLLM validated
     - SGLang: weight sync (``update_weights``) not yet supported on Intel GPU
   * - Trainer backend
     - FSDP, FSDP2
     - Megatron not yet validated
   * - Algorithms
     - GRPO, PPO, SFT
     - Validated on GSM8K / Qwen2.5
   * - Hardware (1-GPU)
     - Validated
     - Arc Pro B60: 51.2 s/step, 148.2 tok/s (batch 16, Qwen2.5-0.5B)
   * - Hardware (2-GPU)
     - Validated
     - Arc Pro B60 ×2: 93.0 s/step at same batch size; scales with larger batch
   * - Multi-node
     - Not yet tested
     - —

Known Limitations
-----------------

**Level Zero VA pressure (2-GPU+).**
Level Zero maps Intel GPU device memory into each process's CPU virtual address
space (~2.2 TB ``VmPeak`` per process). With many colocated Ray workers, this
exhausts kernel page table resources. The test scripts mitigate this with:

- ``RAY_memory_monitor_refresh_ms=0`` — disables Ray's OOM monitor (actual RAM
  usage is normal; the spike is a driver artifact).
- ``RAY_NUM_PRESTART_PYTHON_WORKERS=0`` — reduces idle worker processes.
- vLLM ``uni`` executor forced for TP=1 — avoids spawning a second Level Zero
  context per vLLM server.

**SGLang weight sync.**
``update_weights`` over IPC fails on Intel GPU due to a ``ForkingPickler``
authentication error when crossing Ray actor boundaries. A POSIX shared-memory
workaround exists but has not been upstreamed. Use vLLM as the rollout engine
for now.

**ONEAPI_DEVICE_SELECTOR.**
Do not manually propagate ``ONEAPI_DEVICE_SELECTOR`` to Ray workers. Invalid
values (for example ``level_zero:``) can block oneDNN from finding its OpenCL
device and crash SDPA (oneDNN primitive init). The Docker image includes a
startup guard that repairs empty/invalid values; for device placement use
``ZE_AFFINITY_MASK`` instead.

Example Workflow
-----------------

Prepare data (run once)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

1) GRPO — 1 GPU
~~~~~~~~~~~~~~~~

.. code-block:: bash

    NUM_GPUS=1 bash tests/special_intel_gpu/run_grpo_intel_gpu.sh

Expected output (Qwen2.5-0.5B-Instruct, GSM8K, batch 16, 1 step):

.. code-block:: text

    timing_s/step: ~51  timing_per_token_ms/gen: ~0.35
    perf/throughput: ~148 tok/s

  Verified on this image: the command launches successfully through dataset load,
  FSDP initialization, and vLLM server startup on a 1-GPU run.

2) GRPO — 2 GPUs
~~~~~~~~~~~~~~~~~

.. note::

   The script defaults to the same batch size as the 1-GPU run to keep the
   comparison apples-to-apples. Increase ``data.train_batch_size`` to see
   the full throughput benefit of the second GPU.

.. code-block:: bash

    NUM_GPUS=2 bash tests/special_intel_gpu/run_grpo_intel_gpu.sh

Expected output (same batch size as 1-GPU baseline):

.. code-block:: text

    timing_s/step: ~93  perf/throughput: ~41 tok/s

3) PPO
~~~~~~

The PPO script defaults to 4 GPUs (critic model doubles the memory footprint).
Pass ``NUM_GPUS=2`` to run on a 2× GPU workstation with CPU offload enabled:

.. code-block:: bash

    NUM_GPUS=2 bash tests/special_intel_gpu/run_ppo_intel_gpu.sh

4) SFT
~~~~~~

The SFT script defaults to 4 GPUs. Pass ``NUM_GPUS=2`` for a 2× GPU workstation:

.. code-block:: bash

    NUM_GPUS=2 bash tests/special_intel_gpu/run_sft_intel_gpu.sh

Manual Training Launch
-----------------------

To launch training directly, use the following instructions.

.. code-block:: bash

    export ZE_AFFINITY_MASK=0,1          # select physical device indices
    unset ONEAPI_DEVICE_SELECTOR         # must NOT be set
    export RAY_memory_monitor_refresh_ms=0
    export RAY_NUM_PRESTART_PYTHON_WORKERS=0

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$HOME/data/gsm8k/test.parquet \
        data.train_batch_size=16 \
        data.max_prompt_length=512 \
        data.max_response_length=128 \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=False \
        +actor_rollout_ref.model.override_config.attn_implementation=eager \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.enforce_eager=True \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.total_epochs=1 \
        +ray_kwargs.ray_init.num_gpus=1
