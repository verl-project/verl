Ascend Quick Start Guide
==========================

**Last updated:** 2026/11/06.

.. contents:: Table of Contents
   :local:
   :depth: 2

Key Updates
-----------

- 2026/11/06: The FSDP2 quick-start scripts and the PPO smoke-test script now run on small nodes (e.g. a single A2 card) out of the box via ``NDEVICES_PER_NODE``; added a "Running on a Single Card or a Small Node" section.
- 2026/06/30: Added coverage for four common training and inference backend combinations, making it easier for users to quickly select the appropriate launch script during the quickstart phase.
- 2026/05/13: Separated the quick start and install guidance documents.
- 2025/12/11: verl existing scenarios now support automatic NPU device type detection. GPU scripts running on Ascend no longer need to explicitly set the ``trainer.device=npu`` parameter in principle. New features can still prioritize a specific device by setting ``trainer.device``, with gradual adaptation to automatic detection.

Hardware Support
-----------------

- Atlas 200T A2 Box16
- Atlas 900 A2 PoD
- Atlas 800T A3



Quick Start
---------------------------------

This document targets the Ascend NPU environment and provides a minimal GRPO training validation flow based on GSM8K and Qwen3-0.6B, covering four training and inference backend combinations to help users get started quickly.

Validation scope:

- Environment entry: Whether the verl entry point is available;
- Data reading: Whether the GSM8K dataset is parsed correctly;
- Component initialization: Whether the actor, rollout, and reference workers can be initialized;
- Model inference: Whether the vLLM-Ascend/SGLang rollout can generate outputs;
- End-to-end pipeline: Whether the training flow can complete the first step.

Configuration tips:

- Default configuration: The example scripts all use the Qwen3-0.6B model and the GSM8K dataset.
- Hardware adaptation: Since each A3 card contains 2 dies, running the example requires setting the ``n_gpus_per_node`` parameter to 16 (each A2 card contains 1 die).

Environment Preparation
~~~~~~~~~~~~~~~~~~~~~~~~

Before running the scripts in this document, ensure that the verl Ascend environment has been installed. For installation details, see the :doc:`Ascend Installation Guide <./install_guidance>`.

Weight Preparation
~~~~~~~~~~~~~~~~~~~~

1. Download the Qwen3-0.6B model weights from Hugging Face yourself.

2. The default weight path in the scripts is ``~/models/Qwen/Qwen3-0.6B``. It is recommended to place the weights at this path. If the path differs, modify MODEL_PATH in the script to point to your local path.

Data Preparation
~~~~~~~~~~~~~~~~~

1. Download the original GSM8K dataset from Hugging Face yourself.

2. Run the following command (adjust the command according to your actual dataset path):

   .. code-block:: bash

      python3 examples/data_preprocess/gsm8k.py --local_dataset_path /download/path/hf_data/gsm8k/

   The following files are generated:

      .. code-block:: text

         ~/data/gsm8k/train.parquet
         ~/data/gsm8k/test.parquet

How to Run
~~~~~~~~~~~

1. Enter the project directory: ``cd /your/path/verl``.

2. Enable the CANN environment: Run the following commands. (If you customized the CANN path, adjust the commands according to your actual path.)

   .. code-block:: bash

      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      source /usr/local/Ascend/nnal/atb/set_env.sh

3. Run the script: Four common training and inference backend combinations are currently provided. refer to the table below to select the corresponding script:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 60

   * - Combination
     - Training Backend
     - Rollout Backend
     - How to Run
   * - vLLM + FSDP2
     - FSDP2
     - vLLM-Ascend
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_vllm_ascend.sh
   * - vLLM + Megatron
     - Megatron
     - vLLM-Ascend
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_vllm_ascend.sh
   * - SGLang + FSDP2
     - FSDP2
     - SGLang
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_sglang_ascend.sh
   * - SGLang + Megatron
     - Megatron
     - SGLang
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_sglang_ascend.sh

For detailed parameter descriptions within the scripts, see :doc:`Training Configuration Parameters and Metrics <../dev_guide/model_dev/parameter_and_metrics>`.

For multi-node task launch, see the :doc:`Multi-Machine Task Startup Practice Guide <../model_support/examples/multi-machine_task_startup_practice>`.

Running on a Single Card or a Small Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quick-start scripts assume an 8-device (die) node by default. They also work
on smaller nodes, such as a single Atlas A2 (910B) card, by overriding
``NDEVICES_PER_NODE``. ``ROLLOUT_TP`` is clamped to the device count
automatically, and ``ppo_mini_batch_size`` already defaults to the device count,
so no other overrides are required:

.. code-block:: bash

   # Single A2 card (1 die), smoke run of the FSDP2 + vLLM-Ascend combo
   NDEVICES_PER_NODE=1 SAVE_FREQ=-1 TOTAL_TRAINING_STEPS=1 \
       bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_vllm_ascend.sh

   # Single A3 card (2 dies)
   NDEVICES_PER_NODE=2 SAVE_FREQ=-1 TOTAL_TRAINING_STEPS=1 \
       bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_vllm_ascend.sh

   # PPO smoke test on a single A2 card. The critic adds a second model, so
   # ulysses_sequence_parallel_size must drop to 1, and the default rollout
   # memory utilization (0.8) OOMs once cudagraph capture starts — 0.5 is
   # verified on a single card.
   NDEVICES_PER_NODE=1 SP_SIZE=1 ROLLOUT_GPU_MEM_UTIL=0.5 \
       SAVE_FREQ=-1 TOTAL_TRAINING_STEPS=1 \
       bash tests/special_npu/run_qwen3_06b_ppo.sh

Notes:

- vLLM-Ascend 0.30.x rejects FRACTAL_NZ weights in RL scenarios (weight
  sync after sleep/wake raises an error), so the scripts explicitly pass
  ``additional_config.weight_nz_mode=0`` to the rollout engine.
- The FSDP2 + SGLang script accepts the same ``NDEVICES_PER_NODE`` and
  ``ROLLOUT_TP`` overrides.
- The Megatron combos hardcode ``TP=2, PP=2`` and therefore need at least 4
  devices.
- On a single card, one training step with the default batch (16 prompts x n
  samples) takes noticeably longer. For a quick validation, pass
  ``TOTAL_TRAINING_STEPS=1`` and optionally reduce the batch, e.g.
  ``TRAIN_BATCH_SIZE=8 ROLLOUT_N=4``.

Appendix
-------------------------------------------

SGLang Backend Enablement Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

verl currently parses common inference parameters. For details, see the ``ServerArgs`` initialization parameters in `async_sglang_server.py <https://github.com/verl-project/verl/blob/main/verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_.

Other `SGLang parameters <https://github.com/sgl-project/sglang/blob/v0.5.10/docs/advanced_features/server_arguments.md>`_ can be passed through ``engine_kwargs``.


Converting vLLM Backend Scripts to SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to convert a vLLM backend inference script to SGLang yourself, you need to add or modify the following parameters.

.. code-block:: bash

   # Required
   actor_rollout_ref.rollout.name=sglang \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \

   # Optional
   # Enable inference EP; for detailed usage, see:
   # https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep" \

   # Must be set to True for multi-DP with MoE models (default is False)
   +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False \
