GSM8K Example
=============

Last updated: 03/25/2025.

Introduction
------------

In this example, we train an LLM to tackle the GSM8k task.

Paper: https://arxiv.org/pdf/2110.14168

Dataset: https://huggingface.co/datasets/openai/gsm8k

Note that the original paper mainly focuses on training a verifier (a
reward model) to solve math problems via Best-of-N sampling. In this
example, we train an RLHF agent using a rule-based reward model.

Dataset Introduction
--------------------

GSM8k is a math problem dataset. The prompt is an elementary school
problem. The LLM model is required to answer the math problem.

The training set contains 7473 samples and the test set contains 1319
samples.

**An example**

Prompt

   Katy makes coffee using teaspoons of sugar and cups of water in the
   ratio of 7:13. If she used a total of 120 teaspoons of sugar and cups
   of water, calculate the number of teaspoonfuls of sugar she used.

Solution

   The total ratio representing the ingredients she used to make the
   coffee is 7+13 = <<7+13=20>>20 Since the fraction representing the
   number of teaspoons she used is 7/20, she used 7/20\ *120 =
   <<7/20*\ 120=42>>42 #### 42

Step 1: Prepare dataset
-----------------------

.. code:: bash

   cd examples/data_preprocess
   python3 gsm8k.py --local_save_dir ~/data/gsm8k

Step 2: Download Model
----------------------

There're three ways to prepare the model checkpoints for post-training:

- Download the required models from huggingface or modelscope

.. code:: bash

   hf download deepseek-ai/deepseek-math-7b-instruct --local-dir ~/models/deepseek-math-7b-instruct --local-dir-use-symlinks False
   # or
   modelscope download --model deepseek-ai/deepseek-math-7b-instruct --local_dir ~/models/deepseek-math-7b-instruct

- Already store your store model in the local directory or HDFS path.
- Also, you can directly use the model name in huggingface (e.g.,
  deepseek-ai/deepseek-math-7b-instruct) in
  ``actor_rollout_ref.model.path`` and ``critic.model.path`` field in
  the run script. You can also download models from modelscope by setting environmental variable ``VERL_USE_MODELSCOPE=True``.

Noted that users should prepare checkpoints for actor, critic and reward
model.

[Optional] Step 3: SFT your Model
---------------------------------

We provide a SFT Trainer using PyTorch FSDP in
`sft_trainer.py <https://github.com/verl-project/verl/blob/main/verl/trainer/sft_trainer.py>`_. 
Users can customize their own SFT
script using our FSDP SFT Trainer.

We also provide various training scripts for SFT on GSM8K dataset in `gsm8k sft directory <https://github.com/verl-project/verl/blob/main/examples/sft/gsm8k/>`_.

.. code:: shell

   set -x

   torchrun -m verl.trainer.sft_trainer \
       data.train_files=$HOME/data/gsm8k/train.parquet \
       data.val_files=$HOME/data/gsm8k/test.parquet \
       data.messages_key=messages \
       data.micro_batch_size_per_gpu=8 \
       model.path=deepseek-ai/deepseek-coder-6.7b-instruct \
       trainer.project_name=gsm8k-sft \
       trainer.experiment_name=gsm8k-sft-deepseek-coder-6.7b-instruct \
       trainer.total_epochs=4 \
       trainer.logger='["console","wandb"]'


If you use AMD GPUs (ROCm kernel), you need to add the following environment variables into the run script:

    .. code-block:: bash

        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
        export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES


Step 4: Perform PPO training with your model on GSM8K Dataset
-------------------------------------------------------------

- Prepare your own run.sh script. Here's an example for GSM8k dataset
  and Qwen3-8B model.
- Users could replace the ``data.train_files`` ,\ ``data.val_files``,
  ``actor_rollout_ref.model.path`` and ``critic.model.path`` based on
  their environment.
- See :doc:`config` for detailed explanation of each config field.

**Reward Model/Function**

We use a rule-based reward model. We force the model to produce a final
answer following 4 “#” as shown in the solution. We extract the final
answer from both the solution and model's output using regular
expression matching. We compare them and assign a reward of 1 to correct
answer, 0.1 to incorrect answer and 0 to no answer.

**Training Script**

The training script examples for FSDP and Megatron-LM backends are stored in the ``examples/ppo_trainer`` directory.

.. code:: bash

   cd ../ppo_trainer
   bash run_qwen3_8b_fsdp.sh

``run_qwen3_8b_fsdp.sh`` defaults to ``Qwen/Qwen3-8B`` with FSDP + vLLM
on GSM8K + MATH (override ``MODEL_PATH`` / data paths as needed).
Megatron-LM backend: ``run_qwen3_8b_megatron.sh``.

Key defaults inside ``run_qwen3_8b_fsdp.sh``:

.. code:: bash

   MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
   CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}
   INFER_BACKEND=${INFER_BACKEND:-vllm}
   GSM8K_TRAIN_FILE=${GSM8K_TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
   GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-$HOME/data/gsm8k/test.parquet}
   MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-$HOME/data/math/train.parquet}
   MATH_TEST_FILE=${MATH_TEST_FILE:-$HOME/data/math/test.parquet}


If you use AMD GPUs (ROCm kernel), you need to add the following environment variables into the run script:

    .. code-block:: bash

        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
        export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES

If you encounter any issues in using AMD GPUs running VeRL, feel free to contact me - `Yusheng Su <https://yushengsu-thu.github.io/>`_.