Ascend Quick Start Guide
========================

**Last updated:** 2026/07/14.

Key updates
-----------

- 2026/06/30: Added coverage for four commonly used training and inference backend combinations to help you quickly select an appropriate startup script during the quickstart phase.
- 2026/05/13: Separated the quick start and installation guidance.
- 2025/12/11: Existing verl scenarios support automatic identification of NPU device types. When you run GPU scripts on Ascend, you generally do not need to explicitly set the ``trainer.device=npu`` parameter. You can still prioritize new features by setting ``trainer.device``, gradually adapting to the automatic identification capability.


Contents
--------

- `Hardware Support <#hardware-support>`_
- `Qwen3-0.6B GSM8K GRPO Quick Start <#qwen3-06b-gsm8k-grpo-quick-start>`_
   - `Weight Preparation <#weight-preparation>`_
   - `Data Preparation <#data-preparation>`_
   - `Running Method <#running-method>`_
- `SGLang Backend Enablement Guide <#sglang-backend-enablement-guide>`_
   - `Converting vLLM Backend Scripts to SGLang <#converting-vllm-backend-scripts-to-sglang>`_

Hardware Support
----------------

- Atlas 200T A2 Box16
- Atlas 900 A2 PODc
- Atlas 800T A3



Qwen3-0.6B GSM8K GRPO Quick Start
---------------------------------

This document targets the Ascend NPU environment and provides a minimal GRPO training validation workflow based on GSM8K and Qwen3-0.6B.

This document covers four common training and inference backend combinations to help you quickly select the appropriate startup script during the quickstart phase.

Before you run the scripts in this document, ensure that you have installed the verl Ascend environment.
For details about installing the environment, refer to `Ascend Installation Guide <./install_guidance.rst>`_.

Each A3 device contains 2 dies, and each A2 device contains 1 die. If you run the sample on an A3 machine, set ``n_gpus_per_node`` to 16.

By default, the four scripts use ``Qwen/Qwen3-0.6B`` and the GSM8K dataset for basic pipeline verification.

It is mainly used to check:

- Whether the verl entry point is available;
- Whether the data can be read;
- Whether the actor, rollout, and reference workers can be initialized;
- Whether the vLLM-Ascend/sglang rollout can generate outputs;
- Whether the training pipeline can complete the first step.

Weight Preparation
~~~~~~~~~~~~~~~~~~

Download the model weights from Hugging Face yourself.

The default path for reading weights in the script is ``~/models/Qwen/Qwen3-0.6B``.

Place the weights in this path, or modify MODEL_PATH in the script to point to the local path.


Data preparation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_dataset_path /download/path/hf_data/gsm8k/

You need to download the original gsm8k dataset from Hugging Face.

Generated files:

.. code-block:: text

   ~/data/gsm8k/train.parquet
   ~/data/gsm8k/test.parquet

Running method
~~~~~~~~~~~~~~

The related scripts are located in the ``tests/special_npu/quick_start/`` directory.

First, navigate to the verl directory: ``cd /your/path/verl``

Enable the CANN environment: If you have customized the CANN path, modify the following enable command according to your custom path.

.. code-block:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh

Quick Start provides four common training and rollout backend combinations. You can select the corresponding script based on the training backend and the rollout backend.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 60

   * - Combination
     - Training backend
     - Rollout backend
     - Execution method
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

For detailed descriptions of the parameters in the script, see `Training Configuration Parameters and Metrics <../dev_guide/model_dev/parameter_and_metrics.md>`_

For details on multi-node task startup, refer to the `Multi-machine Task Startup Guide <../model_support/examples/multi-machine_task_startup_practice.rst>`_.

SGLang backend enablement instructions
-------------------------------------------

verl currently parses common inference parameters. For details, see the ``ServerArgs`` initialization parameters in `async_sglang_server.py <../../../../verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_.

You can pass other `SGLang parameters <https://github.com/sgl-project/sglang/blob/v0.5.10/docs/advanced_features/server_arguments.md>`_ using ``engine_kwargs``.

Convert vLLM backend scripts to SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to manually convert the vLLM backend inference script to SGLang, add or modify the following parameters.

.. code-block:: bash

   # Required
   actor_rollout_ref.rollout.name=sglang \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \

   # Optional
   # Enable inference EP. For detailed usage, see:
   # https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep" \

   # Must be set to True when using multiple DP for MoE models
   +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False \

   # chunked_prefill is disabled by default
   +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1

