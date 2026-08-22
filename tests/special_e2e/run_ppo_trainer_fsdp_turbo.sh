#!/usr/bin/env bash
# dependency: GPU vllm==0.18.0, transformers@<cc7ab9be>, fsdp_turbo
# FSDP-Turbo e2e smoke test for Qwen3.5-0.8B (dense) on GPU.

set -xeuo pipefail

INFER_BACKEND=${INFER_BACKEND:-vllm}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-0.8B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
TRAIN_FILE=${TRAIN_FILE:-${HOME}/data/geo3k/train.parquet}
TEST_FILE=${TEST_FILE:-${HOME}/data/geo3k/test.parquet}

GEN_TP=${GEN_TP:-2}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_N=${ROLLOUT_N:-2}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}

PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3_5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Qwen3_5-2B-FSDP-Turbo}
SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}" .sh)"
LOG_DIR=${LOG_DIR:-/tmp/verl_logs/$SCRIPT_NAME}
mkdir -p $LOG_DIR
rm -rf $LOG_DIR/$SCRIPT_NAME.log

n_devices_per_node=8

########################### shared turbo config values ###########################
ACTOR_TURBO="actor_rollout_ref.actor.fsdp_config.turbo_config"
REF_TURBO="actor_rollout_ref.ref.fsdp_config.turbo_config"

FSDP_APPLY_MODULES='{model.visual.blocks.\{*\}:{},'\
'model.visual.patch_embed:{},'\
'model.visual.pos_embed:{},'\
'model.language_model.embed_tokens:{},'\
'model.language_model.layers.\{*\}:{},'\
'lm_head:{}}'

HOOK_MODULES="['model.language_model.layers.{*}']"
RECOMPUTE_PLAN="['model.language_model.layers.{*}','model.visual.blocks.{*}']"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.image_key=images
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.optimizer=AdamW
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.strategy=fsdp_turbo
    "${ACTOR_TURBO}.distributed.fully_shard_parallel_size=${FSDP_SIZE}"
    "${ACTOR_TURBO}.distributed.ulysses_parallel_size=${SP_SIZE}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${ACTOR_TURBO}.memory.recompute=True"
    "+${ACTOR_TURBO}.memory.recompute_plan=${RECOMPUTE_PLAN}"
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.strategy=fsdp_turbo
    "${REF_TURBO}.distributed.fully_shard_parallel_size=${FSDP_SIZE}"
    "${REF_TURBO}.distributed.ulysses_parallel_size=${SP_SIZE}"
    "+${REF_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${REF_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${REF_TURBO}.memory.recompute=True"
    "+${REF_TURBO}.memory.recompute_plan=${RECOMPUTE_PLAN}"
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,8,12,16,24,32,48,56,64]"
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=1
    trainer.balance_batch=False
    trainer.resume_from_path=checkpoints/
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=15
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" | tee $LOG_DIR/$SCRIPT_NAME.log
