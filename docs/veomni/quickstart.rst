.. _quickstart:

==================================================================
Quickstart: PPO training on GSM8K dataset with VeOmniEngine
==================================================================

Step 1: Install VeOmni
----------------------------

We'll use the stable v0.1.4 tag as an example.

.. code-block:: bash

   pip3 install git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.4


Step 1: Prepare the dataset
----------------------------

We preprocess the dataset in parquet format so that it contains necessary fields for computing RL rewards and is faster to read.

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

Step 2: Download Qwen3-30B-A3B for post-training
---------------------------------------------------


Step 3: Preprocess the model
------------------------------------

We merge individual MoE expert weights into stacked tensors for efficient loading.

.. code-block:: bash

   python3 scripts/veomni/moe_merge.py --raw_hf_path <input_checkpoint> --merge_hf_path <output_dir>

Step 4: Run PPO training


**Training Script**

Now let's run PPO training with the dataset and model above.


Set the ``data.train_files`` ,\ ``data.val_files``, ``actor_rollout_ref.model.path`` and ``critic.model.path`` based on your dataset and model names or paths.

.. code-block:: bash

    #!/usr/bin/env bash
    set -xeuo pipefail

    MODEL_PATH=${MODEL_PATH:-${HOME}/path/to/preprocessed_model}
    #huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

    TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
    VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}
    VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
    NUM_GPUS=${NUM_GPUS:-8}
    FSDP_SIZE=${FSDP_SIZE:-4}
    SP_SIZE=${SP_SIZE:-2}
    EP_SIZE=${EP_SIZE:-1}
    VERL_EXP_NAME=${VERL_EXP_NAME:-${MODEL_PATH}-function-reward-minimal-fsdp-size${FSDP_SIZE}}

    python3 -m verl.trainer.main_ppo \
        model_engine=veomni \
        algorithm.adv_estimator=grpo \
        data.train_files="${TRAIN_FILES}" \
        data.val_files="${VAL_FILES}" \
        data.train_batch_size=16 \
        data.max_prompt_length=512 \
        data.max_response_length=128 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.veomni.param_offload=True \
        actor_rollout_ref.actor.veomni.optimizer_offload=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.actor.veomni.fsdp_size="${FSDP_SIZE}" \
        actor_rollout_ref.actor.veomni.ulysses_parallel_size="${SP_SIZE}" \
        actor_rollout_ref.actor.veomni.expert_parallel_size="${EP_SIZE}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.veomni.param_offload=True \
        actor_rollout_ref.ref.use_torch_compile=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.n=2 \
        actor_rollout_ref.ref.veomni.optimizer_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.use_legacy_worker_impl=disable \
        trainer.critic_warmup=0 \
        trainer.logger=console \
        trainer.project_name='verl_veomni_test' \
        trainer.experiment_name="${VERL_EXP_NAME}" \
        trainer.n_gpus_per_node="${NUM_GPUS}" \
        trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=500 $@



