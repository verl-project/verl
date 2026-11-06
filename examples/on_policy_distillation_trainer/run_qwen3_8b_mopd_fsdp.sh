#!/usr/bin/env bash
# On-policy distillation | multi-teacher (gsm8k text + geo3k VL) | vLLM rollout | FSDP training | NVIDIA GPUs
#
# By default this script enables the multi-teacher shared GPU group
# (TEACHER_SHARE_GPU_GROUP=True): BOTH teachers are placed on the SAME GPUs instead of
# disjoint sub-pools, so the teacher pool only needs to fit ONE teacher
# (TEACHER_WORLD_SIZE = teacher_tp * num_replicas). Teacher engines sleep when idle
# (vLLM sleep level 1: weights offloaded to CPU, KV cache freed) and are woken on
# demand by the TeacherSleepController. With both teachers at
# gpu_memory_utilization=0.4 both can stay awake simultaneously, so
# distillation.max_awake_teachers is left unset (null = unlimited).
# Set TEACHER_SHARE_GPU_GROUP=False to fall back to disjoint per-teacher pools
# (teacher pool = sum of both teachers' world sizes).

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
GSM8K_TEACHER_MODEL=${GSM8K_TEACHER_MODEL:-Qwen/Qwen3-32B}
GEO3K_TEACHER_MODEL=${GEO3K_TEACHER_MODEL:-Qwen/Qwen3-VL-32B-Instruct}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Per-teacher replicas; total teacher GPUs depend on TEACHER_SHARE_GPU_GROUP (see header).
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_NUM_REPLICAS_GSM8K=${TEACHER_NUM_REPLICAS_GSM8K:-1}
TEACHER_NUM_REPLICAS_GEO3K=${TEACHER_NUM_REPLICAS_GEO3K:-1}
teacher_tp=${TEACHER_TP:-2}

TEACHER_SHARE_GPU_GROUP=${TEACHER_SHARE_GPU_GROUP:-True}
if [ "$TEACHER_SHARE_GPU_GROUP" = "True" ]; then
    # Shared mode places both teachers on the same GPUs, so per-teacher world sizes
    # must be equal; the teacher pool only needs to fit one teacher.
    if [ "$TEACHER_NUM_REPLICAS_GSM8K" != "$TEACHER_NUM_REPLICAS_GEO3K" ]; then
        echo "ERROR: TEACHER_SHARE_GPU_GROUP=True requires equal per-teacher world sizes, but " >&2
        echo "TEACHER_NUM_REPLICAS_GSM8K ($TEACHER_NUM_REPLICAS_GSM8K) != TEACHER_NUM_REPLICAS_GEO3K ($TEACHER_NUM_REPLICAS_GEO3K)." >&2
        echo "Set them equal, or set TEACHER_SHARE_GPU_GROUP=False to use disjoint teacher pools." >&2
        exit 1
    fi
    TEACHER_WORLD_SIZE=$(( teacher_tp * TEACHER_NUM_REPLICAS_GSM8K ))
else
    # Disjoint pools: total teacher GPUs = sum(num_replicas) * teacher_tp
    TEACHER_WORLD_SIZE=$(( (TEACHER_NUM_REPLICAS_GSM8K + TEACHER_NUM_REPLICAS_GEO3K) * teacher_tp ))
fi

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_distill_mopd_gsm8k_geo3k}
experiment_name=${EXPERIMENT_NAME:-qwen3_vl_8b_from_qwen3_32b_and_qwen3_vl_32b_mopd_vllm_fsdp}
# ---- end user-adjustable ----

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
geo3k_train=$HOME/data/geo3k/train.parquet
geo3k_test=$HOME/data/geo3k/test.parquet

train_files="['$gsm8k_train', '$geo3k_train']"
val_files="['$gsm8k_test', '$geo3k_test']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    data.image_key=images
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# Multi-teacher: one teacher per dataset, routed by the sample's `data_source` value.
# Use `+distillation.teacher_models.<name>.*` to add named teachers; the default `teacher_model`
# entry is silently popped when other teacher entries are added.
EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    # --- gsm8k teacher (text) ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="$GSM8K_TEACHER_MODEL"
    +distillation.teacher_models.gsm8k.num_replicas=${TEACHER_NUM_REPLICAS_GSM8K}
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.gsm8k.inference.max_model_len=${max_num_tokens}
    # --- geo3k teacher (VL) ---
    +distillation.teacher_models.geo3k.key="hiyouga/geometry3k"
    +distillation.teacher_models.geo3k.model_path="$GEO3K_TEACHER_MODEL"
    +distillation.teacher_models.geo3k.num_replicas=${TEACHER_NUM_REPLICAS_GEO3K}
    +distillation.teacher_models.geo3k.inference.name=vllm
    +distillation.teacher_models.geo3k.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.geo3k.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.geo3k.inference.max_model_len=${max_num_tokens}
    # --- loss ---
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

if [ "$TEACHER_SHARE_GPU_GROUP" = "True" ]; then
    EXTRA+=(distillation.share_gpu_group=True)
fi

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the vllm × fsdp extras of the committed uv.lock;
# NPU falls back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
