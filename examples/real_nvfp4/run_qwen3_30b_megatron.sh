#!/usr/bin/env bash
set -euo pipefail

# GRPO training on Qwen3-30B-A3B with native W4A4 routed-expert MLPs,
# per-token rollout activation scaling, and BF16 boundary-layer carve-outs.

readonly PRECISION_MODE=${PRECISION_MODE:-real_nvfp4}
case "$PRECISION_MODE" in
  bf16|real_nvfp4) ;;
  *) echo "PRECISION_MODE must be bf16 or real_nvfp4" >&2; exit 2 ;;
esac
readonly TRAIN_PROMPT_BSZ=32
readonly N_RESP_PER_PROMPT=16
readonly PPO_MINI_BATCH_SIZE=32
readonly MAX_RESPONSE_LENGTH=20480
readonly MAX_TOKEN_LEN=21504
readonly MAX_NUM_BATCHED_TOKENS=32768
readonly MAX_NUM_SEQS=256
readonly AGENT_NUM_WORKERS=8

readonly WORKING_DIR=${WORKING_DIR:-$PWD}
readonly RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}
readonly RUNTIME_ENV=${RUNTIME_ENV:-$WORKING_DIR/examples/real_nvfp4/runtime_env.yaml}
readonly NNODES=${NNODES:-8}
readonly N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
readonly PROJECT_NAME=${PROJECT_NAME:-verl-nvfp4}
readonly EXP_NAME=${EXP_NAME:?set EXP_NAME to a new W&B/checkpoint run id}
readonly RAY_DATA_HOME=${RAY_DATA_HOME:-$WORKING_DIR}
readonly MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to your shared model directory}
readonly TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE to your preprocessed training parquet}
readonly TEST_FILE=${TEST_FILE:?set TEST_FILE to your preprocessed validation parquet}
readonly CKPTS_DIR=${CKPTS_DIR:-$RAY_DATA_HOME/checkpoints/$PROJECT_NAME/$EXP_NAME}
readonly TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
readonly RESUME_MODE=${RESUME_MODE:-disable}
readonly RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

[[ -f "$MODEL_PATH/config.json" ]]
[[ -f "$TRAIN_FILE" && -f "$TEST_FILE" && -f "$RUNTIME_ENV" ]]
[[ "$RESUME_MODE" = disable || "$RESUME_MODE" = auto || "$RESUME_MODE" = resume_path ]]
if [[ "$RESUME_MODE" = resume_path ]]; then
  [[ -d "$RESUME_FROM_PATH" && "$RESUME_FROM_PATH" = *global_step_* ]]
else
  [[ -z "$RESUME_FROM_PATH" ]]
fi

# Transformer Engine and FlashInfer read their NVFP4 settings from the worker
# environment, which RUNTIME_ENV provides.

DATA=(
  data.train_files="$TRAIN_FILE"
  data.val_files="$TEST_FILE"
  data.prompt_key=prompt
  data.return_raw_chat=True
  data.truncation=left
  data.filter_overlong_prompts=True
  data.filter_overlong_prompts_workers=1
  data.max_prompt_length=1024
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.train_batch_size="$TRAIN_PROMPT_BSZ"
)

ALGORITHM=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.rollout_correction.rollout_is=token
  algorithm.rollout_correction.rollout_is_threshold=2.0
  algorithm.rollout_correction.rollout_is_batch_normalize=False
  algorithm.rollout_correction.rollout_rs=null
)

MODEL=(
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.actor.optim.lr_decay_style=constant
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.betas='[0.9,0.999]'
  actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=True
  actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.optimizer_offload=True
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.actor.megatron.context_parallel_size=1
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=4
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.actor.megatron.sequence_parallel=False
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.router_replay.mode=R3
  +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.expert_parallel_size=1
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_model_len="$MAX_TOKEN_LEN"
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS"
  actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.n="$N_RESP_PER_PROMPT"
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS"
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.top_k=-1
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.enable_rollout_routing_replay=True
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=False
)

FORWARD_ONLY=(
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN"
)

REWARD=(reward.reward_manager.name=naive)

TRAINER=(
  trainer.logger='["console","wandb"]'
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$EXP_NAME"
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
  trainer.nnodes="$NNODES"
  trainer.val_before_train=False
  trainer.test_freq=10
  trainer.save_freq="${CHECKPOINT_SAVE_FREQ:-10}"
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
  trainer.total_epochs=100
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS"
  trainer.default_local_dir="$CKPTS_DIR"
  trainer.resume_mode="$RESUME_MODE"
  trainer.log_val_generations=2
)
if [[ "$RESUME_MODE" = resume_path ]]; then
  TRAINER+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi

# Routed-expert MLPs run NVFP4 except in the first 2 and last 4 decoder layers,
# which stay BF16 in both training and rollout.
PRECISION=(actor_rollout_ref.actor.megatron.real_nvfp4.enable=False)
if [[ "$PRECISION_MODE" = real_nvfp4 ]]; then
  PRECISION=(
    actor_rollout_ref.actor.megatron.real_nvfp4.enable=True
    actor_rollout_ref.actor.megatron.real_nvfp4.num_layers_at_start_in_bf16=2
    actor_rollout_ref.actor.megatron.real_nvfp4.num_layers_at_end_in_bf16=4
  )
fi

HYDRA_ARGS=(
  --config-name=ppo_megatron_trainer \
  "${DATA[@]}" \
  "${ALGORITHM[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${FORWARD_ONLY[@]}" \
  "${REWARD[@]}" \
  "${TRAINER[@]}" \
  "${PRECISION[@]}"
)

if [[ "${CONFIG_ONLY:-0}" = 1 ]]; then
  python3 -m verl.trainer.main_ppo --cfg job --resolve "${HYDRA_ARGS[@]}"
  exit 0
fi

export RAY_ADDRESS
ray job submit --runtime-env="$RUNTIME_ENV" -- \
  python3 -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}"
