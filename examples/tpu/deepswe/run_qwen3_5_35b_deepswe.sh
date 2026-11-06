#!/usr/bin/env bash
# DeepSWE | Qwen3.5-35B-A3B | R2E-Gym + ReAct (uni-agent) | TorchTitan trainer + vLLM rollout | TPU
#
# Preprocess R2E-Gym dataset beforehand with uni-agent:
#   python3 -m uni_agent.tasks.r2e_gym.preprocess --local-save-dir "${RAY_DATA_HOME}/data/r2e_gym_easy"

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RAY_DATA_HOME="${RAY_DATA_HOME:-/data}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
export MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/${MODEL_NAME}}"
export TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/r2e_gym_easy/train.parquet}"
export TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/r2e_gym_easy/test.parquet}"
export PROJECT_NAME="${PROJECT_NAME:-verl-tpu-deepswe}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_35b_deepswe_r2e_gym}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-251}"
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-57344}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_TOKEN_LEN_PER_DEVICE="${MAX_TOKEN_LEN_PER_DEVICE:-${MAX_MODEL_LEN}}"
export PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"

UNI_AGENT_HOME="${UNI_AGENT_HOME:-/workspace/uni-agent}"
TASK_CONFIG="${TASK_CONFIG:-${UNI_AGENT_HOME}/examples/quickstart/training/task_config_react.yaml}"
GATEWAY_COUNT="${GATEWAY_COUNT:-4}"
CONCURRENCY="${CONCURRENCY:-32}"
DISPATCH_MODE="${DISPATCH_MODE:-inline_async}"
TRAJECTORY_SELECTION="${TRAJECTORY_SELECTION:-all}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-/tmp/uni_agent_logs/${EXPERIMENT_NAME}}"
MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-False}"

AGENT_OVERRIDES=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES:-verl_hardware_plugin}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_NAMESPACE='${SANDBOX_NAMESPACE:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_IMAGE_PREFIX='${SANDBOX_IMAGE_PREFIX:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.NODE_SELECTOR_KEY='${NODE_SELECTOR_KEY:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.NODE_SELECTOR_VAL='${NODE_SELECTOR_VAL:-}'"
  "++actor_rollout_ref.rollout.multi_turn.format=qwen3_coder"
  "++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
  "++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}"
  "++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR}"
  "++actor_rollout_ref.rollout.custom.agent_framework.allowed_request_sampling_param_keys=[temperature,top_p,top_k]"
  "++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=${DISPATCH_MODE}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=${TRAJECTORY_SELECTION}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${MODEL_NAME}"
  "reward.reward_manager.name=dapo"
  "reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner"
  "reward.custom_reward_function.name=score_from_runner_result"
)

if [[ "${TPU_ACCELERATOR_TYPE:-tpu7x}" == "v5p" ]]; then
  V5P_LIBTPU_INIT_ARGS="${V5P_LIBTPU_INIT_ARGS:---slicebuilder_use_insecure_grpc=true --undefok=slicebuilder_use_insecure_grpc --xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_scoped_vmem_limit_kib=32768 --xla_tpu_enable_bitcast_reduce_fusion_graduation=false --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false --xla_tpu_bf16_emission_mode=PROMOTE_F32}"
  AGENT_OVERRIDES+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.TPU_ACCELERATOR_TYPE='v5p'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_TPU_TOPOLOGY='${TORCH_TPU_TOPOLOGY:-2,2,4}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.USE_MOE_SPARSE_CORE='${USE_MOE_SPARSE_CORE:-0}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RAY_EXTRA_ENV_VARS_TO_COPY='LIBTPU_INIT_ARGS,USE_MOE_SPARSE_CORE,TORCH_TPU_TOPOLOGY,TPU_ACCELERATOR_TYPE'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.LIBTPU_INIT_ARGS='${V5P_LIBTPU_INIT_ARGS}'"
  )
fi

exec bash "${SCRIPT_DIR}/../grpo/run_qwen3_5_torchtitan.sh" \
    algorithm.adv_estimator=rloo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=seq_mean_k1 \
    "algorithm.rollout_correction.rollout_rs_threshold='0.999_1.002'" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.clip_grad=0.125 \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.999]" \
    actor_rollout_ref.actor.optim.min_lr_factor=1.0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    ++data.max_length="${MAX_MODEL_LEN}" \
    ++data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    data.return_raw_chat=True \
    data.truncation=error \
    actor_rollout_ref.rollout.mode=async \
    "${AGENT_OVERRIDES[@]}" \
    trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD:-1}" \
    trainer.v1.sampler.max_off_policy_strategy=wait \
    "$@"
