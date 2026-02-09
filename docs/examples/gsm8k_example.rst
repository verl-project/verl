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
  See examples/ppo_trainer/run_deepseek7b_llm_modelscope.sh for example.

Noted that users should prepare checkpoints for actor, critic and reward
model.

[Optional] Step 3: SFT your Model
---------------------------------

We provide a SFT Trainer using PyTorch FSDP in
`fsdp_sft_trainer.py <https://github.com/volcengine/verl/blob/main/verl/trainer/fsdp_sft_trainer.py>`_. 
Users can customize their own SFT
script using our FSDP SFT Trainer.

We also provide various training scripts for SFT on GSM8K dataset in `gsm8k sft directory <https://github.com/volcengine/verl/blob/main/examples/sft/gsm8k/>`_.

.. code:: shell

   set -x

   torchrun -m verl.trainer.fsdp_sft_trainer \
       data.train_files=$HOME/data/gsm8k/train.parquet \
       data.val_files=$HOME/data/gsm8k/test.parquet \
       data.prompt_key=question \
       data.response_key=answer \
       data.micro_batch_size_per_gpu=8 \
       model.partial_pretrain=deepseek-ai/deepseek-coder-6.7b-instruct \
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
  and deepseek-llm-7b-chat model.
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

The training script example for FSDP and Megatron-LM backend are stored in examples/ppo_trainer directory.

.. code:: bash

   cd ../ppo_trainer
   bash run_deepseek7b_llm.sh

The script of run_deepseek7b_llm.sh

.. code:: bash

   set -x

   python3 -m verl.trainer.main_ppo \
      data.train_files=$HOME/data/gsm8k/train.parquet \
      data.val_files=$HOME/data/gsm8k/test.parquet \
      data.train_batch_size=1024 \
      data.max_prompt_length=512 \
      data.max_response_length=512 \
      actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=256 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      critic.optim.lr=1e-5 \
      critic.model.use_remove_padding=True \
      critic.model.path=deepseek-ai/deepseek-llm-7b-chat \
      critic.model.enable_gradient_checkpointing=True \
      critic.ppo_micro_batch_size_per_gpu=32 \
      critic.model.fsdp_config.param_offload=False \
      critic.model.fsdp_config.optimizer_offload=False \
      algorithm.kl_ctrl.kl_coef=0.001 \
      trainer.critic_warmup=0 \
      trainer.logger='["console","wandb"]' \
      trainer.project_name='verl_example_gsm8k' \
      trainer.experiment_name='deepseek_llm_7b_function_rm' \
      trainer.n_gpus_per_node=8 \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=1 \
      trainer.total_epochs=15 $@


If you use AMD GPUs (ROCm kernel), you need to add the following environment variables into the run script:

    .. code-block:: bash

        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
        export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES

If you encounter any issues in using AMD GPUs running VeRL, feel free to contact me - `Yusheng Su <https://yushengsu-thu.github.io/>`_.