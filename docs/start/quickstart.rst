.. _quickstart:

=========================================================
Quickstart: PPO training on GSM8K dataset
=========================================================

Post-train a LLM using GSM8K dataset.

Introduction
------------

.. _hf_dataset_gsm8k: https://huggingface.co/datasets/openai/gsm8k

In this example, we train an LLM to tackle the `GSM8k <hf_dataset_gsm8k>`_ task with function-based rewards. [1]_

Prerequisite:

- the latest version of ``verl`` and its dependencies installed following the installation guide. Using the docker image is recommended.

- a GPU with at least 24 GB HBM


Dataset Introduction
--------------------

GSM8k is a math problem dataset. The prompt is an elementary school
problem. The LLM model is asked to solve the math problem. Below is an example:

Prompt

   Katy makes coffee using teaspoons of sugar and cups of water in the
   ratio of 7:13. If she used a total of 120 teaspoons of sugar and cups
   of water, calculate the number of teaspoonfuls of sugar she used.

Solution

   The total ratio representing the ingredients she used to make the
   coffee is 7+13 = <<7+13=20>>20 Since the fraction representing the
   number of teaspoons she used is 7/20, she used 7/20\ *120 =
   <<7/20*\ 120=42>>42 #### 42

Step 1: Prepare the dataset
----------------------------

We preprocess the dataset in parquet format so that (1) it contains necessary fields for computing RL rewards and (2) is faster to read.

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

Step 2: Download a model for post-training
-------------------------------------------

In this example, we start with the ``Qwen2.5-0.5B-Instruct`` model.

If you want to perform SFT before RL, refer to the :doc:`Complete GSM8K Example<../examples/gsm8k_example>`, the `sft directory <https://github.com/volcengine/verl/blob/main/examples/sft/gsm8k>`_ and `SFT Trainer <https://github.com/volcengine/verl/blob/main/verl/trainer/fsdp_sft_trainer.py>`_ for further details.

.. code-block:: bash

   python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"

Step 3: Perform PPO training with the instruct model
----------------------------------------------------------------------

**Reward Model/Function**

We use a pre-defined rule-based reward model. We force the model to produce a final
answer following 4 “#” as shown in the solution. We extract the final
answer from both the solution and model's output using regular
expression matching. We assign a reward of 1 to correct
answer, 0.0 to incorrect answer and 0 to no answer. 

For more details, please refer to `verl/utils/reward_score/gsm8k.py <https://github.com/volcengine/verl/blob/v0.4.1/verl/utils/reward_score/gsm8k.py>`_.

**Training Script**

Now let's run PPO training with the dataset and model above. [2]_


Set the ``data.train_files`` ,\ ``data.val_files``, ``actor_rollout_ref.model.path`` and ``critic.model.path`` based on your dataset and model names or paths.
You may set ``VERL_USE_MODELSCOPE=True`` to download models from `modelscope <https://www.modelscope.cn>`_ instead of `huggingface <https://huggingface.co>`_.

.. code-block:: bash

   PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log

You are expected to see the following logs, indicating training in progress. The key metric ``val/test_score/openai/gsm8k`` is computed every ``trainer.test_freq`` steps:

.. code-block:: bash

    step:0 - timing/gen:21.470 - timing/ref:4.360 - timing/values:5.800 - actor/reward_kl_penalty:0.000 - actor/reward_kl_penalty_coeff:0.001 - timing/adv:0.109 - timing/update_critic:15.664 - critic/vf_loss:14.947 - critic/vf_clipfrac:0.000 - critic/vpred_mean:-2.056 - critic/grad_norm:1023.278 - critic/lr(1e-4):0.100 - timing/update_actor:20.314 - actor/entropy_loss:0.433 - actor/pg_loss:-0.005 - actor/pg_clipfrac:0.000 - actor/ppo_kl:0.000 - actor/grad_norm:1.992 - actor/lr(1e-4):0.010 - critic/score/mean:0.004 - critic/score/max:1.000 - critic/score/min:0.000 - critic/rewards/mean:0.004 - critic/rewards/max:1.000 - critic/rewards/min:0.000 - critic/advantages/mean:-0.000 - critic/advantages/max:2.360 - critic/advantages/min:-2.280 - critic/returns/mean:0.003 - critic/returns/max:0.000 - critic/returns/min:0.000 - critic/values/mean:-2.045 - critic/values/max:9.500 - critic/values/min:-14.000 - response_length/mean:239.133 - response_length/max:256.000 - response_length/min:77.000 - prompt_length/mean:104.883 - prompt_length/max:175.000 - prompt_length/min:68.000
    step:1 - timing/gen:23.020 - timing/ref:4.322 - timing/values:5.953 - actor/reward_kl_penalty:0.000 - actor/reward_kl_penalty:0.001 - timing/adv:0.118 - timing/update_critic:15.646 - critic/vf_loss:18.472 - critic/vf_clipfrac:0.384 - critic/vpred_mean:1.038 - critic/grad_norm:942.924 - critic/lr(1e-4):0.100 - timing/update_actor:20.526 - actor/entropy_loss:0.440 - actor/pg_loss:0.000 - actor/pg_clipfrac:0.002 - actor/ppo_kl:0.000 - actor/grad_norm:2.060 - actor/lr(1e-4):0.010 - critic/score/mean:0.000 - critic/score/max:0.000 - critic/score/min:0.000 - critic/rewards/mean:0.000 - critic/rewards/max:0.000 - critic/rewards/min:0.000 - critic/advantages/mean:0.000 - critic/advantages/max:2.702 - critic/advantages/min:-2.616 - critic/returns/mean:0.000 - critic/returns/max:0.000 - critic/returns/min:0.000 - critic/values/mean:-2.280 - critic/values/max:11.000 - critic/values/min:-16.000 - response_length/mean:232.242 - response_length/max:256.000 - response_length/min:91.000 - prompt_length/mean:102.398 - prompt_length/max:185.000 - prompt_length/min:70.000

Checkout ``Algorithm Baselines`` page for full training and validation logs for reference.

The checkpoint is saved at the following dir by default: ``checkpoints/${trainer.project_name}/${trainer.experiment_name}``. You can merge the saved checkpoints to huggingface model using ``verl.model_merger`` module, for example:

.. code-block:: bash

    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_step_1/actor \
        --target_dir checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_step_1/actor/huggingface

For more details about checkpoint and model merging, please refer to :ref:`checkpoint-page`.

To enable ``wandb`` for experiment tracking, set the following configs:

.. code-block:: bash

    trainer.logger='["console","wandb"]' \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \

If you encounter out of memory issues with HBM less than 32GB, enable the following configs would help:

.. code-block:: bash

    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_micro_batch_size_per_gpu=1 \

For the full set of configs, please refer to :ref:`config-explain-page` for detailed explanation and performance tuning.


.. [1] The original paper (https://arxiv.org/pdf/2110.14168) mainly focuses on training a verifier (a reward model) to solve math problems via Best-of-N sampling. In this example, we train an RL agent using a rule-based reward model.
.. [2] More training script examples for FSDP and Megatron-LM backend are stored in `examples/ppo_trainer <https://github.com/volcengine/verl/tree/main/examples/ppo_trainer>`_ directory.
