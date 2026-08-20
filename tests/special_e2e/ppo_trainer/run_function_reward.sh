#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-512}

ENGINE=${ENGINE:-vllm}
if [ "$ENGINE" = "vllm" ]; then
    export VLLM_USE_V1=1
fi
ROLLOUT_MODE="async"

RETURN_RAW_CHAT="True"
SKIP_TOKENIZER_INIT="True"

GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
ACTOR_FSDP_PARAM_OFFLOAD=${ACTOR_FSDP_PARAM_OFFLOAD:-True}
ACTOR_FSDP_OPTIMIZER_OFFLOAD=${ACTOR_FSDP_OPTIMIZER_OFFLOAD:-True}
REF_FSDP_PARAM_OFFLOAD=${REF_FSDP_PARAM_OFFLOAD:-True}
RM_PAD=${RM_PAD:-True}
FUSED_KERNELS=${FUSED_KERNELS:-False}
FUSED_KERNEL_BACKEND=${FUSED_KERNEL_BACKEND:-torch} # or 'triton' for triton backend
ADV_ESTIMATOR=${ADV_ESTIMATOR:-gae}
LOSS_MODE=${LOSS_MODE:-vanilla}
USE_KL=${USE_KL:-False}
CUSTOM_REWARD_FN=${CUSTOM_REWARD_FN:-False}
CUSTOM_REWARD_FN_FILE=${CUSTOM_REWARD_FN_FILE:-}
DATA_SHUFFLE=${DATA_SHUFFLE:-True}
DATA_SEED=${DATA_SEED:-null}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}
ROLLOUT_FULL_DETERMINISM=${ROLLOUT_FULL_DETERMINISM:-False}
ROLLOUT_SCHEDULING_POLICY=${ROLLOUT_SCHEDULING_POLICY:-fcfs}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ACTOR_FULL_DETERMINISM=${ACTOR_FULL_DETERMINISM:-False}
REF_FULL_DETERMINISM=${REF_FULL_DETERMINISM:-False}
CRITIC_FULL_DETERMINISM=${CRITIC_FULL_DETERMINISM:-False}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True} # For vLLM VLM placeholder issue: https://github.com/vllm-project/vllm/issues/15185
STRATEGY=${STRATEGY:-fsdp}
# LoRA config
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-${LORA_RANK}}
LORA_TARGET=${LORA_TARGET:-"all-linear"}
LORA_EXCLUDE=${LORA_EXCLUDE:-"DONT_EXCLUDE"}
USE_SHM=${USE_SHM:-False}
LOAD_FORMAT=${LOAD_FORMAT:-dummy}
LAYERED_SUMMON=${LAYERED_SUMMON:-False}
# Validation
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TEST_FREQ=${TEST_FREQ:--1}
# Save & Resume
RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:--1}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}
OUTPUT_FILE=${OUTPUT_FILE:-$(pwd)/output.txt}
KEEP_OUTPUT_FILE=${KEEP_OUTPUT_FILE:-False}

# whether to save hf_model
SAVE_HF_MODEL=${SAVE_HF_MODEL:-False}
FSDP_SIZE=${FSDP_SIZE:--1}
SP_SIZE=${SP_SIZE:-1}

if [ "${SAVE_HF_MODEL}" = "True" ]; then
    CHECKPOINT_CONTENTS="['model','hf_model','optimizer','extra']"
else
    CHECKPOINT_CONTENTS="['model','optimizer','extra']"
fi

train_traj_micro_bsz_per_gpu=${TRAIN_TRAJ_MICRO_BSZ_PER_GPU:-2} # b
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4} # g

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * 1)) # b * n
train_traj_mini_bsz=$((train_traj_micro_bsz * 2)) # 2 * b * n
train_prompt_mini_bsz=$((train_traj_mini_bsz * 2)) # 2 * b * n / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

reward_fn_name=null
reward_fn_file_path=null
generated_reward_fn_file=False
output_file="${OUTPUT_FILE}"
if [ "${CUSTOM_REWARD_FN}" = "True" ]; then
    reward_fn_name="my_reward_function"
    if [ -n "${CUSTOM_REWARD_FN_FILE}" ]; then
        reward_fn_file_path="${CUSTOM_REWARD_FN_FILE}"
    else
        generated_reward_fn_file=True
        reward_fn_file_path="$(pwd)/my_reward_function.py"
        rm -rf "${reward_fn_file_path}"
        cat <<EOF > "$reward_fn_file_path"
def ${reward_fn_name}(data_source, solution_str, ground_truth, extra_info=None):
    print(f"Congratulations!!! You have called ${reward_fn_name} successfully!!!")
    return 0.1
EOF
    fi

    rm -rf "${output_file}"
fi

exp_name="${VERL_EXP_NAME:-$(basename "${MODEL_ID,,}")-function-reward-minimal}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.shuffle="${DATA_SHUFFLE}" \
    data.seed="${DATA_SEED}" \
    data.train_batch_size="${train_prompt_bsz}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    data.return_raw_chat=${RETURN_RAW_CHAT} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_shm=${USE_SHM} \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.target_modules=${LORA_TARGET} \
    actor_rollout_ref.model.exclude_modules=${LORA_EXCLUDE} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding="${RM_PAD}" \
    actor_rollout_ref.model.use_fused_kernels=${FUSED_KERNELS} \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_KERNEL_BACKEND} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.strategy=${STRATEGY} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_FSDP_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_FSDP_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.full_determinism="${ACTOR_FULL_DETERMINISM}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.checkpoint.save_contents=${CHECKPOINT_CONTENTS} \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL}" \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
    actor_rollout_ref.rollout.seed="${ROLLOUT_SEED}" \
    actor_rollout_ref.rollout.full_determinism="${ROLLOUT_FULL_DETERMINISM}" \
    actor_rollout_ref.rollout.scheduling_policy="${ROLLOUT_SCHEDULING_POLICY}" \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.load_format=${LOAD_FORMAT} \
    actor_rollout_ref.rollout.layered_summon=${LAYERED_SUMMON} \
    actor_rollout_ref.rollout.skip_tokenizer_init="${SKIP_TOKENIZER_INIT}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_FSDP_PARAM_OFFLOAD}" \
    actor_rollout_ref.ref.fsdp_config.full_determinism="${REF_FULL_DETERMINISM}" \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding="${RM_PAD}" \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True \
    critic.fsdp.full_determinism="${CRITIC_FULL_DETERMINISM}" \
    reward.custom_reward_function.path="${reward_fn_file_path}"\
    reward.custom_reward_function.name="${reward_fn_name}"\
    algorithm.use_kl_in_reward="${USE_KL}" \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl-test' \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.total_epochs=2 \
    trainer.device=cuda \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" $@ \
    2>&1 | tee "${output_file}"

if [ "${CUSTOM_REWARD_FN}" = "True" ]; then
    python3 tests/special_e2e/check_custom_rwd_fn.py --output_file="${output_file}"
    check_exit_code=$?
    if [ "${generated_reward_fn_file}" = "True" ]; then
        rm -rf "${reward_fn_file_path}"
    fi
    if [ "${KEEP_OUTPUT_FILE}" != "True" ]; then
        rm -rf "${output_file}"
    fi
    # Return the exit code of check_custom_rwd_fn.py if it fails
    if [ $check_exit_code -ne 0 ]; then
        exit $check_exit_code
    fi
fi
