#!/usr/bin/env bash
set -xeuo pipefail

# E2E smoke test for the V1 trainer `fully_async` (streaming) mode.
#
# `fully_async` builds on `separate_async`: a STANDALONE rollout cluster (its own GPUs) serves
# generation while the trainer runs on separate GPUs, and weights are synced via the NCCL
# checkpoint engine. The difference vs separate_async is that an autonomous background feeder
# continuously streams prompts into TransferQueue (bounded by a staleness / in-flight budget)
# instead of feeding exactly one batch per training step.
#
# GPU allocation (separate, NOT colocated):
#   - trainer.n_gpus_per_node          -> training GPUs
#   - actor_rollout_ref.rollout.n_gpus_per_node -> standalone rollout GPUs
#   total GPUs required = training + rollout. Minimal smoke test: 2 GPUs (1 + 1).

export VERL_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1

# Split GPUs between training and standalone rollout.
NUM_GPUS=${NUM_GPUS:-2}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-$((NUM_GPUS / 2))}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-$((NUM_GPUS - N_GPUS_TRAINING))}

# Defaults point at the shared /efs mount (visible on all cluster nodes). Override via env.
MODEL_PATH=${MODEL_PATH:-/efs/data/rl/v1/models/Qwen/Qwen2.5-0.5B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-/efs/data/rl/v1/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-/efs/data/rl/v1/gsm8k/test.parquet}

rollout_name=${ROLLOUT_NAME:-vllm}

# Algorithm
adv_estimator=grpo
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4}

# separate_async / fully_async require train_batch_size == ppo_mini_batch_size.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}
train_prompt_mini_bsz=${train_prompt_bsz}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

# Streaming / off-policy knobs.
parameter_sync_step=${PARAMETER_SYNC_STEP:-4}
staleness_threshold=${STALENESS_THRESHOLD:-1}
# Streaming relies on the feeder's in-flight budget to BOUND staleness, and on truncated
# importance sampling (TIS, calculate_log_probs=True) to CORRECT the residual off-policyness —
# NOT on the replay buffer hard-dropping samples. So the trainer-side staleness gate is off by
# default (strategy "none"). Set MAX_OFF_POLICY_STRATEGY=drop (+ a small MAX_OFF_POLICY_THRESHOLD)
# only to exercise the hard-drop path.
max_off_policy_strategy=${MAX_OFF_POLICY_STRATEGY:-none}
max_off_policy_threshold=${MAX_OFF_POLICY_THRESHOLD:-$((staleness_threshold + 1))}

total_training_steps=${TOTAL_TRAINING_STEPS:-20}

exp_name="$(basename "${MODEL_PATH,,}")-v1-fully-async-streaming"

echo "Running V1 fully_async (streaming) trainer"
echo "Training GPUs: ${N_GPUS_TRAINING} | Standalone rollout GPUs: ${N_GPUS_ROLLOUT}"

# Submit as a Ray job so the head node's working_dir (this checkout) is shipped to ALL nodes
# for this run and prepended to sys.path — edit code only on the head, no `git pull` on every
# worker. Override RAY_DASHBOARD / VERL_HOME if your layout differs.
RAY_DASHBOARD=${RAY_DASHBOARD:-http://127.0.0.1:8265}
VERL_HOME=${VERL_HOME:-${HOME}/verl}
runtime_env_json=$(cat <<JSON
{"working_dir": "${VERL_HOME}", "excludes": ["/.git", "/data", "*.parquet", "*.whl", "*.pt", "*.safetensors", "__pycache__"], "env_vars": {"VERL_LOGGING_LEVEL": "${VERL_LOGGING_LEVEL:-INFO}", "VLLM_USE_V1": "1"}}
JSON
)

ray job submit \
    --address "${RAY_DASHBOARD}" \
    --runtime-env-json "${runtime_env_json}" \
    -- python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=fully_async \
    trainer.v1.fully_async.num_warmup_batches=0 \
    trainer.v1.fully_async.parameter_sync_step=${parameter_sync_step} \
    trainer.v1.fully_async.staleness_threshold=${staleness_threshold} \
    trainer.v1.fully_async.feeder_poll_interval=1.0 \
    trainer.v1.sampler.max_off_policy_strategy=${max_off_policy_strategy} \
    trainer.v1.sampler.max_off_policy_threshold=${max_off_policy_threshold} \
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
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=${N_GPUS_ROLLOUT} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend='nccl' \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    reward.reward_model.enable=False \
    trainer.logger='["console"]' \
    trainer.project_name='verl-test-v1-fully-async' \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${N_GPUS_TRAINING} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${total_training_steps} \
    "$@"

echo "V1 fully_async streaming E2E test completed successfully"
