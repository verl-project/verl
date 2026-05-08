set -euo pipefail

# Required assets before cluster submission:
# 1. MODEL_PATH points to a local Qwen3-Omni Thinker checkpoint directory.
# 2. TRAIN_FILE / TEST_FILE point to parquet files whose prompts use <audio> placeholders
#    and whose audio payload column is configured by data.audio_key (default: audios).
# 3. The runtime environment must provide transformers with Qwen3-Omni support,
#    vLLM with multimodal audio input support, Ray CLI, and qwen_omni_utils when
#    use_audio_in_video=True is desired.

if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then
  ENGINE=$1
  shift
else
  ENGINE=${ENGINE:-vllm}
fi

: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-Omni Thinker checkpoint directory.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the converted OmniInstruct training parquet.}"
: "${TEST_FILE:?Set TEST_FILE to the converted OmniInstruct validation parquet.}"

PROJECT_NAME=${PROJECT_NAME:-"GSPO-Qwen3-Omni-Thinker"}
EXP_NAME=${EXP_NAME:-"GSPO-Qwen3-Omni-Thinker-FSDP-${ENGINE}"}

# Use Hydra interpolation so a ``trainer.experiment_name=...`` CLI override
# automatically produces a per-experiment checkpoint directory. Without this,
# parallel A/B runs would all share the same CKPTS_DIR and auto-resume would
# try to load an unrelated run's state_dict (and fail on mesh mismatches such
# as flat FSDP mesh (N,) vs HSDP mesh (nnodes, fsdp_size)).
CKPTS_DIR=${CKPTS_DIR:-"checkpoints/\${trainer.project_name}/\${trainer.experiment_name}"}

USE_AUDIO_IN_VIDEO=${USE_AUDIO_IN_VIDEO:-false}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-120}

VAL_FILE="${TEST_FILE}"

# Cluster geometry. Defaults reproduce the original 8-GPU setup; override when
# splitting the cluster across two parallel runs (e.g. NNODES=4
# N_GPUS_PER_NODE=8 for a 32-GPU job).
NNODES=${NNODES:-2}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
# Keep FSDP sharding intra-node by default. With NNODES>1 this gives HSDP
# (FSDP within a node, DP across nodes) which keeps all-gather/reduce-scatter
# on NVLink instead of IB.
FSDP_SIZE=${FSDP_SIZE:-${N_GPUS_PER_NODE}}
# Rollout TP must fit within a node (NVLink). 4 is a sweet spot for 30B-A3B.
GEN_TP=${GEN_TP:-4}
SP_SIZE=${SP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

# Batch geometry. Override these values to match the target cluster size and
# memory budget.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-24576}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
LR=${LR:-5e-7}
KL_LOSS_COEF=${KL_LOSS_COEF:-1e-3}
GRAD_CLIP=${GRAD_CLIP:-0.5}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=64 \
    data.val_max_samples=64 \
    data.truncation='error' \
    data.audio_key=audios \
    data.shuffle=False \
    +data.mm_processor_kwargs.use_audio_in_video=${USE_AUDIO_IN_VIDEO} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.resume_mode=disable \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.balance_batch=True \
    trainer.val_before_train=False \
    trainer.total_epochs=10 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    "$@"
