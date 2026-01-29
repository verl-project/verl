#!/bin/bash

YOUR_PROJECT_NAME=r1-verl-grpo-upstream
YOUR_RUN_NAME=r1-training_grpo-upstream
# export HYDRA_FULL_ERROR=1
# export FSDP_VERBOSE=1

TRAIN_PATH="$(cd "$(dirname "$0")"; pwd)"
export TRAIN_PATH=$TRAIN_PATH
export HF_HOME="/wekafs/0_public/huggingface"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "TRAIN_PATH: $TRAIN_PATH"

# [ray] < 2.45.0
#export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
# [ray] >= 2.45.0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 # Patch with https://github.com/ray-project/ray/pull/52794
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 # Patch with https://github.com/ray-project/ray/pull/52794
export VLLM_USE_FLASHINFER_MOE_FP8=0
# TODO: aiter will fail on Qwen3-30B-A3B-Instruct model
export VLLM_ROCM_USE_AITER=0

# choose your model
# MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
# MODEL_PATH=Qwen/Qwen2-7B-Instruct
MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507

# preprocess data (need to be done only once)
# python3 ../../examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
# python3 -c "import transformers; transformers.pipeline('text-generation', model='$MODEL_PATH')"

export GPUS_PER_NODE=8
export ENGINE=vllm #sglang

ray stop --force
# ray start --head --node-ip-address=127.0.0.1 --port=6379


python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_PATH}/data/gsm8k/train.parquet" \
    data.val_files="${TRAIN_PATH}/data/gsm8k/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    2>&1 | tee log.txt
