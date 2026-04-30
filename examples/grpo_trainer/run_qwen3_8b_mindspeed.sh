#!/usr/bin/env bash
# GRPO | text | MindSpeed-LLM training | Ascend NPU
#
# Set INFER_BACKEND=sglang (default).

set -xeuo pipefail

# ---- user-adjustable ----
INFER_BACKEND=${INFER_BACKEND:-sglang}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
micro_bsz=${MICRO_BSZ:-1}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_tp=${ACTOR_TP:-4}
actor_pp=${ACTOR_PP:-4}
actor_cp=${ACTOR_CP:-1}
all_offload=${ALL_OFFLOAD:-True}

rollout_tp=${ROLLOUT_TP:-4}
rollout_dp=${ROLLOUT_DP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
rollout_n=${ROLLOUT_N:-8}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-verl_grpo_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_mindspeed}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/verl/ckpts/${project_name}/${experiment_name}"}
# ---- end user-adjustable ----

# ---- system defaults (normally leave as-is) ----
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export DISABLE_L2_CACHE=1
export TASK_QUEUE_ENABLE=1
# For CANN 8.5.0+, when using mbridge:
export HCCL_OP_EXPANSION_MODE=AIV
# ---- end system defaults ----

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet

train_files="['$gsm8k_train', '$math_train']"
val_files="['$gsm8k_test', '$math_test']"

max_model_len=$((max_prompt_length + max_response_length))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation=left
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_model_len}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.mindspeed.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.mindspeed.param_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.optimizer_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.grad_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True
    actor_rollout_ref.actor.mindspeed.llm_kwargs.spec='[mindspeed_llm.tasks.models.spec.qwen3_spec, layer_spec]'
    actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_length=${max_model_len}
    actor_rollout_ref.actor.mindspeed.llm_kwargs.micro_batch_size=${micro_bsz}
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_query_groups=8
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_method=uniform
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_granularity=full
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_num_layers=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.overlap_grad_reduce=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.overlap_param_gather=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_model_len}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.data_parallel_size=${rollout_dp}
    actor_rollout_ref.rollout.enforce_eager=False
)

REF=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_model_len}
    actor_rollout_ref.ref.mindspeed.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.mindspeed.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.mindspeed.context_parallel_size=${actor_cp}
    actor_rollout_ref.ref.mindspeed.param_offload=${all_offload}
    actor_rollout_ref.ref.mindspeed.use_mbridge=True
    actor_rollout_ref.ref.mindspeed.vanilla_mbridge=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NPUS_PER_NODE}
    trainer.val_before_train=False
    trainer.test_freq=${test_freq}
    trainer.save_freq=${save_freq}
    trainer.total_epochs=${total_epochs}
    trainer.default_local_dir="${CKPTS_DIR}"
)

EXTRA=(
    model_engine=mindspeed
)

if [ "${INFER_BACKEND}" = sglang ]; then
    EXTRA+=(
        +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=ascend
        +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
