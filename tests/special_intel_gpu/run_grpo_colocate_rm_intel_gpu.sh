#!/usr/bin/env bash
# E2E GRPO test on Intel GPU with a COLOCATED reward model — mirrors
# tests/special_e2e/run_v1_colocate_async_disrm.sh.
#
# Unlike run_grpo_intel_gpu.sh (which goes through RolloutReplica.init_hybrid()
# for the policy rollout), this script also enables a discriminative reward
# model with reward.reward_model.enable_resource_pool=False, so the reward
# model's rollout replica shares the same GPUs as training and is initialized
# via RolloutReplica.init_colocated() (verl/workers/rollout/replica.py).
#
# Use this to exercise the init_colocated()/init_standalone() device_name path
# on XPU (the buggy `device_name="cuda" if not is_torch_npu_available(...) ...`
# ternary), which run_grpo_intel_gpu.sh never reaches.
#
# GPU allocation: all GPUs are shared between training, rollout and the
# reward model (colocate); 2 GPUs is enough for this smoke test.
#
# Usage:
#   NUM_GPUS=2 bash tests/special_intel_gpu/run_grpo_colocate_rm_intel_gpu.sh

set -x

NUM_GPUS=${NUM_GPUS:-2}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}
RM_MODEL_ID=${RM_MODEL_ID:-Skywork/Skywork-Reward-V2-Llama-3.2-1B}
RM_MODEL_PATH=${RM_MODEL_PATH:-${RM_MODEL_ID}}

adv_estimator=grpo
n_resp_per_prompt=4
num_reward_workers=${NUM_REWARD_WORKERS:-4}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-${train_prompt_bsz}}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-128}
# Reward-model rollout must fit the full chat (prompt + response + RM template overhead).
rm_prompt_length=$(( max_prompt_length + max_response_length + 512 ))

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=1 \
    transfer_queue.enable=True \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward.num_workers=${num_reward_workers} \
    reward.reward_manager.name=dapo \
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool=False \
    reward.reward_model.model_path="${RM_MODEL_PATH}" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=1 \
    reward.reward_model.rollout.gpu_memory_utilization=0.4 \
    reward.reward_model.rollout.enforce_eager=True \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.skip_tokenizer_init=False \
    reward.reward_model.rollout.prompt_length=${rm_prompt_length} \
    reward.reward_model.rollout.response_length=${max_response_length} \
    trainer.logger=console \
    trainer.project_name='verl_intel_gpu_grpo_colocate_rm_e2e' \
    trainer.experiment_name='qwen2_5_05b_intel_gpu_grpo_colocate_rm' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    +ray_kwargs.ray_init.num_gpus=${NUM_GPUS} $@
