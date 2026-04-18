Megatron-FSDP Example
========================

Last updated: 04/08/2026.

Introduction
------------

In this example, we run SFT and RL training with Megatron-FSDP:

- Runtime image: ``verlai/verl:vllm011.dev7``

Step 1: Prepare code
--------------------

Use the tested PR branches for ``verl``, ``Megatron-LM``, and ``Megatron-Bridge``:

.. code:: bash

   cd /root

   # 1) verl
   git clone https://github.com/verl-project/verl.git
   cd /root/verl
   git fetch origin pull/5423/head:pr-5423
   git checkout pr-5423

   # 2) Megatron-LM
   cd /root
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd /root/Megatron-LM
   git fetch origin pull/3191/head:pr-3191
   git checkout pr-3191

   # 3) Megatron-Bridge
   cd /root
   git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
   cd /root/Megatron-Bridge
   git fetch origin pull/1910/head:pr-1910
   git checkout pr-1910


Step 2: Install dependencies and set environment
------------------------------------------------

.. code:: bash

   cd /root/verl
   pip3 install --no-deps -e .[test]
   pip3 install "nvidia-modelopt[torch]>=0.37.0" math-verify transformers==4.57.1
   export PYTHONPATH=/root/Megatron-LM:/root/Megatron-Bridge/src:$PYTHONPATH

   unset CUDA_DEVICE_MAX_CONNECTIONS
   ray stop --force

Step 3: Prepare datasets
------------------------

.. code:: bash

   cd /root/verl

   # GSM8K
   python3 examples/data_preprocess/gsm8k.py \
     --local_save_dir ~/data/gsm8k

   # MATH
   python3 examples/data_preprocess/math_dataset.py \
     --local_save_dir ~/data/math

   # Check generated parquet files
   ls -lh ~/data/gsm8k/train.parquet ~/data/gsm8k/test.parquet
   ls -lh ~/data/math/train.parquet ~/data/math/test.parquet

Step 4: Run Megatron-FSDP SFT
----------------------------

Before launch, check and update key fields `MODEL_PATH` and `SAVE_PATH` in the script.

.. code:: bash

   bash verl/examples/sft/gsm8k/run_qwen_megatron_fsdp.sh

Step 5: Run Megatron-FSDP RL
----------------------------

Before launch, check and update key fields in
``examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh``:

- ``actor_rollout_ref.model.path``: model name or local model path.
- ``train_files`` / ``test_files``: parquet paths for GSM8K and MATH.
- ``trainer.n_gpus_per_node`` and ``trainer.nnodes``: hardware topology.
- ``trainer.project_name`` and ``trainer.experiment_name``: experiment identifiers.

Then run:

.. code:: bash

   bash examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh

The script launches RL training and enables Megatron-FSDP with:

- ``actor_rollout_ref.actor.megatron.use_mbridge=True``
- ``actor_rollout_ref.actor.megatron.vanilla_mbridge=False``
- ``actor_rollout_ref.actor.megatron.use_megatron_fsdp=True``