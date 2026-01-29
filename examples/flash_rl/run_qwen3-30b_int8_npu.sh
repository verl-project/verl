#!/usr/bin/env bash
set -xeuo pipefail

RAY_DATA_PATH=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_PATH}/models/Qwen3-30B-A3B"}
QUANT_PATH=${QUANT_PATH:-"${RAY_DATA_PATH}/models/Qwen3-30B-A3B-w8a8"}
PROFILE_PATH=${PROFILE_PATH:-"${RAY_DATA_PATH}/profile.30b.pt"}
CONFIG_PATH=${CONFIG_PATH:-"${RAY_DATA_PATH}/.flashrl_config.30b.yaml"}

if ! command -v flashrl &> /dev/null
then
    pip install flash-llm-rl # need to be installed in all nodes in multi-node training
fi

# manually add 'import flash_rl' in 'verl/verl/__init__.py'
if ! grep -q "import flash_rl" "${RAY_DATA_PATH}/verl/__init__.py"; then
    echo "Adding 'import flash_rl' to verl/verl/__init__.py"
    sed -i '1i import flash_rl' "${RAY_DATA_PATH}/verl/__init__.py"
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Profile file not found at ${PROFILE_PATH}. Running profiling and setup..."
    flashrl profile -m ${MODEL_PATH} -q ${QUANT_PATH} -o ${PROFILE_PATH} --fn int8
    flashrl setup -m ${QUANT_PATH} -p ${PROFILE_PATH} --fn int8 -o ${CONFIG_PATH}
else
    echo "Profile file found at ${PROFILE_PATH}. Skipping profiling and setup."
fi
# (Optional) conduct rollout generation in 16bits and 8bits in a hybrid manner across DP workers
# flashrl setup -a --fn bf16 -o ${CONFIG_PATH}

flashrl cleanup

export VERL_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export FLASHRL_LOGGING_LEVEL=DEBUG
export FLASHRL_CONFIG=${CONFIG_PATH}
export FLASHRL_LMHEAD_FP32=1

# HCCL related configuration
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export P2P_HCCL_BUFFSIZE=30
export HCCL_BUFFSIZE=300
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

export RAY_DEDUP_LOGS=1
export HYDRA_FULL_ERROR=1

# For dispatch optimization, set to 1 for graph mode, 2 for non-graph mode
export TASK_QUEUE_ENABLE=2  
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
export PYTHONUNBUFFERED=1

project_name='GRPO'
exp_name='GRPO-Qwen3-30B'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 30))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=32
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=8
max_num_seqs=1024
train_prompt_mini_bsz=32

# Ray
NNODES=1
# Paths
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/gsm8k/test.parquet"}
LOG_PATH="./logs"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=16
use_dynamic_bsz=True
log_prob_micro_batch_size_per_gpu=8
ppo_micro_batch_size_per_gpu=8
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True
gen_tp=4
enable_chunked_prefill=True

# Importance Sampling (IS) weights configuration
rollout_is="sequence"                     # Self-normalized sequence-level IS
rollout_is_threshold=2.0                  # Upper threshold for IS weights
rollout_is_batch_normalize="true"         # Self-normalization (mean=1.0)

# Rejection Sampling (RS) configuration
rollout_rs="null"                         # No rejection sampling for basic RLOO
rollout_rs_threshold="null"               # RS upper threshold
rollout_rs_threshold_lower="null"         # RS lower threshold

# Veto mechanism (optional, independent of IS/RS)
rollout_token_veto_threshold="null"       # Per-token veto threshold (null to disable)

# Policy Gradient loss mode (bypass mode with policy gradient loss, no PPO clipping)
bypass_mode="true"     # Required for policy gradient mode
use_policy_gradient="true"        # Use policy gradient loss (works with IS/RS/both)

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    +data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    +algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    +algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    +algorithm.rollout_correction.use_policy_gradient=${use_policy_gradient} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    ++actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    reward_model.reward_manager=naive \
    +reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    +reward_model.overlong_buffer.len=${overlong_buffer_len} \
    +reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.device='npu' $@

