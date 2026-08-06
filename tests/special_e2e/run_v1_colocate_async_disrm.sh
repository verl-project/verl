#!/usr/bin/env bash
set -xeuo pipefail

# E2E regression test for the V1 trainer (colocate_async) with a COLOCATED discriminative
# reward model (disRM). This exercises PPOTrainer._compute_reward_colocate, which is the
# colocated-RM path shared by all V1 trainers (sync / colocate_async / separate_async).
#
# Unlike run_fully_async_policy_genrm.sh (which uses a STANDALONE GenRM on its own GPUs),
# here the reward model is colocated with the actor/rollout on the same GPUs
# (reward.reward_model.enable_resource_pool=False). The rollout replicas are slept after
# sampling so the reward model can reuse their GPU memory for scoring.
#
# GPU allocation: all GPUs are shared between training, rollout and the reward model
# (colocate); a single node with >=2 GPUs is sufficient for the smoke test.

export VERL_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1

NUM_GPUS=${NUM_GPUS:-2}

# Model paths (default to the conventional local cache used by other verl tests).
MODEL_PATH=${MODEL_PATH:-${HOME}/models/Qwen/Qwen2.5-0.5B-Instruct}
# Discriminative reward model (sequence classification head), same family used by
# tests/special_e2e/run_ppo_trainer_megatron.sh.
RM_MODEL_PATH=${RM_MODEL_PATH:-${HOME}/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B}

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

rollout_name=${ROLLOUT_NAME:-vllm}

# Algorithm parameters
adv_estimator=grpo
n_resp_per_prompt=4

# Keep the batch divisible by reward.num_workers (compute_rm_score chunks across workers).
num_reward_workers=${NUM_REWARD_WORKERS:-4}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}          # 8 prompts x 4 responses = 32 rows, divisible by 4 workers
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-${train_prompt_bsz}}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

# Reward-model rollout must fit the full chat (prompt + response + RM template overhead).
rm_prompt_length=$(( max_prompt_length + max_response_length + 512 ))

exp_name="$(basename "${MODEL_PATH,,}")-v1-colocate-async-disrm-minimal"

echo "Running V1 colocate_async trainer with COLOCATED disRM"
echo "Total GPUs: ${NUM_GPUS} (shared by training / rollout / reward model)"

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=1 \
    transfer_queue.enable=True \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward.num_workers=${num_reward_workers} \
    reward.reward_manager.name=dapo \
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool=False \
    reward.reward_model.model_path="${RM_MODEL_PATH}" \
    reward.reward_model.rollout.name=${rollout_name} \
    reward.reward_model.rollout.tensor_model_parallel_size=1 \
    reward.reward_model.rollout.gpu_memory_utilization=0.6 \
    reward.reward_model.rollout.free_cache_engine=True \
    reward.reward_model.rollout.skip_tokenizer_init=False \
    reward.reward_model.rollout.prompt_length=${rm_prompt_length} \
    reward.reward_model.rollout.response_length=${max_response_length} \
    trainer.logger='["console"]' \
    trainer.project_name='verl-test-v1-colocate-async-disrm' \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    "$@"

echo "V1 colocate_async + colocated disRM E2E test completed successfully"
