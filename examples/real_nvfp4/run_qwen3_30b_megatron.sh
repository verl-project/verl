#!/usr/bin/env bash
set -euo pipefail

# Matched BF16 vs real W4A4 experiment on Qwen3-30B-A3B. Real W4A4 is
# routed-expert MLPs outside BF16 carve-outs, per-token rollout activation,
# R3-on, and none of the three optional stability losses.

readonly PRECISION_MODE=${PRECISION_MODE:-real_nvfp4}
case "$PRECISION_MODE" in
  bf16|real_nvfp4) ;;
  *) echo "PRECISION_MODE must be bf16 or real_nvfp4" >&2; exit 2 ;;
esac
readonly RUN_PROFILE=${RUN_PROFILE:-formal}
case "$RUN_PROFILE" in
  formal)
    readonly EXPECTED_NNODES=8
    readonly TRAIN_PROMPT_BSZ=32
    readonly N_RESP_PER_PROMPT=16
    readonly PPO_MINI_BATCH_SIZE=32
    readonly MAX_RESPONSE_LENGTH=20480
    readonly MAX_TOKEN_LEN=21504
    readonly MAX_NUM_BATCHED_TOKENS=32768
    readonly MAX_NUM_SEQS=256
    readonly AGENT_NUM_WORKERS=8
    ;;
  smoke)
    readonly EXPECTED_NNODES=1
    # The expanded rollout batch must contain at least one item per EP/DP
    # partition.  EP=4 with 2 responses therefore needs 2 prompts.
    readonly TRAIN_PROMPT_BSZ=2
    readonly N_RESP_PER_PROMPT=2
    readonly PPO_MINI_BATCH_SIZE=2
    readonly MAX_RESPONSE_LENGTH=1024
    readonly MAX_TOKEN_LEN=2048
    readonly MAX_NUM_BATCHED_TOKENS=2048
    # The reduced smoke profile is not the formal capacity-validation contract.
    readonly MAX_NUM_SEQS=128
    readonly AGENT_NUM_WORKERS=2
    ;;
  *) echo "RUN_PROFILE must be formal or smoke" >&2; exit 2 ;;
esac

readonly WORKING_DIR=${WORKING_DIR:-$PWD}
readonly RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}
readonly RUNTIME_ENV=${RUNTIME_ENV:-$WORKING_DIR/examples/real_nvfp4/runtime_env.yaml}
readonly NNODES=${NNODES:-8}
readonly N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
readonly PROJECT_NAME=${PROJECT_NAME:-DAPO-NVFP4-QAT}
readonly EXP_NAME=${EXP_NAME:?set EXP_NAME to a new W&B/checkpoint run id}
readonly RAY_DATA_HOME=${RAY_DATA_HOME:-$WORKING_DIR}
readonly MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to your shared model directory}
readonly TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE to your shared dataset file}
readonly TEST_FILE=${TEST_FILE:?set TEST_FILE to your shared dataset file}
readonly CKPTS_DIR=${CKPTS_DIR:-$RAY_DATA_HOME/checkpoints/$PROJECT_NAME/$EXP_NAME}
readonly TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
# Token-level truncated importance sampling caps each importance weight at 2.
# Dynamic sampling refills generation batches until enough nonconstant-reward
# prompt groups are available, up to MAX_GEN_BATCHES.
# Keep the same first/last BF16 layer carve-outs in training and rollout.
readonly FIRST_LAST_BF16=${FIRST_LAST_BF16:-True}
readonly BF16_LAYERS_AT_START=${BF16_LAYERS_AT_START:-2}
readonly BF16_LAYERS_AT_END=${BF16_LAYERS_AT_END:-4}
readonly GEN_PROMPT_BSZ_MULT=${GEN_PROMPT_BSZ_MULT:-2}
readonly FILTER_GROUPS=${FILTER_GROUPS:-True}
readonly MAX_GEN_BATCHES=${MAX_GEN_BATCHES:-10}
readonly GEN_PROMPT_BSZ=$((TRAIN_PROMPT_BSZ * GEN_PROMPT_BSZ_MULT))
readonly ROLLOUT_IS=${ROLLOUT_IS:-token}
readonly STRICT_MINERVA=${STRICT_MINERVA:-1}
# Keep FlashInfer autotuning disabled for the matched BF16/W4A4 recipe.
# Overlong punishment uses a configurable response-length buffer and factor.
readonly FLASHINFER_AUTOTUNE=${FLASHINFER_AUTOTUNE:-False}
readonly OVERLONG_PENALTY=${OVERLONG_PENALTY:-True}
readonly OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-512}
readonly OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
if [[ "$OVERLONG_PENALTY" = True && ( "$OVERLONG_BUFFER_LEN" -le 0 ) ]]; then
  echo "OVERLONG_PENALTY=True requires OVERLONG_BUFFER_LEN>0" >&2; exit 2
fi
case "$ROLLOUT_IS" in token|sequence|null) ;; *) echo "ROLLOUT_IS must be token|sequence|null" >&2; exit 2 ;; esac
case "$STRICT_MINERVA" in 0|1) ;; *) echo "STRICT_MINERVA must be 0 or 1" >&2; exit 2 ;; esac
readonly RESUME_MODE=${RESUME_MODE:-disable}
readonly RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
readonly VERL_WANDB_RUN_ID=${VERL_WANDB_RUN_ID:-}
readonly VERL_WANDB_RESUME=${VERL_WANDB_RESUME:-}
# The carve-out lives in the per-module MCore recipe as well as in Megatron's
# first_last_layers_bf16 flag: the flag skips the FP4 autocast for those layers
# and the recipe gives their MLP the BF16 config, and verl refuses to build if
# the two disagree.
if [[ "$FIRST_LAST_BF16" = True ]]; then
  readonly DEFAULT_TE_PRECISION_CONFIG=$WORKING_DIR/examples/real_nvfp4/config/attn_bf16_mlp_nvfp4_first${BF16_LAYERS_AT_START}_last${BF16_LAYERS_AT_END}.yaml
else
  readonly DEFAULT_TE_PRECISION_CONFIG=$WORKING_DIR/examples/real_nvfp4/config/attn_bf16_mlp_nvfp4.yaml
fi
readonly TE_PRECISION_CONFIG=${TE_PRECISION_CONFIG:-$DEFAULT_TE_PRECISION_CONFIG}

[[ -f "$MODEL_PATH/config.json" ]]
[[ -f "$TRAIN_FILE" && -f "$TEST_FILE" && -f "$RUNTIME_ENV" ]]
[[ "$NNODES" = "$EXPECTED_NNODES" && "$N_GPUS_PER_NODE" = 4 ]]
[[ "$RESUME_MODE" = disable || "$RESUME_MODE" = auto || "$RESUME_MODE" = resume_path ]]
if [[ "$RESUME_MODE" = resume_path ]]; then
  [[ -d "$RESUME_FROM_PATH" && "$RESUME_FROM_PATH" = *global_step_* ]]
else
  [[ -z "$RESUME_FROM_PATH" ]]
fi
if [[ -n "$VERL_WANDB_RUN_ID" ]]; then
  [[ "$VERL_WANDB_RUN_ID" != *['/\\#?%:']* ]]
  [[ "$VERL_WANDB_RESUME" = allow || "$VERL_WANDB_RESUME" = must || "$VERL_WANDB_RESUME" = never ]]
else
  [[ -z "$VERL_WANDB_RESUME" ]]
fi

export VLLM_USE_V1=1
export VERL_LOGGING_LEVEL=INFO
export NVTE_BACKWARD_OVERRIDE=dequantized
export NVTE_NVFP4_ROW_SCALED_ACTIVATION=1
export NVTE_NVFP4_DISABLE_RHT=1
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1
export NVTE_NVFP4_4OVER6=none
export NVTE_NVFP4_4OVER6_E4M3_USE_256=all
export NVTE_NVFP4_4OVER6_ERR_MODE=MAE
export FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1
export TRTLLM_DISABLE_FP4_QUANT_FAST_MATH=1
export VERL_MATH_DAPO_STRICT_MINERVA="$STRICT_MINERVA"

DATA=(
  data.train_files="$TRAIN_FILE"
  data.val_files="$TEST_FILE"
  "+data.boxed_answer_files=['$TEST_FILE']"
  data.prompt_key=prompt
  data.return_raw_chat=True
  data.truncation=left
  data.filter_overlong_prompts=True
  data.filter_overlong_prompts_workers=1
  data.custom_cls.path="$WORKING_DIR/examples/real_nvfp4/slime_math_dataset.py"
  data.custom_cls.name=SlimeMathDataset
  data.max_prompt_length=1024
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.train_batch_size="$TRAIN_PROMPT_BSZ"
  data.gen_batch_size="$GEN_PROMPT_BSZ"
)

ALGORITHM=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.filter_groups.enable="$FILTER_GROUPS"
  algorithm.filter_groups.metric=acc
  algorithm.filter_groups.max_num_gen_batches="$MAX_GEN_BATCHES"
  algorithm.rollout_correction.rollout_is="$ROLLOUT_IS"
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
  actor_rollout_ref.actor.megatron.vanilla_mbridge=False
  actor_rollout_ref.actor.megatron.use_megatron_fsdp=False
  actor_rollout_ref.actor.megatron.router_replay.mode=R3
  actor_rollout_ref.actor.megatron.qat.enable=False
  +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.first_last_layers_bf16="$FIRST_LAST_BF16"
  +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_at_start_in_bf16="$BF16_LAYERS_AT_START"
  +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_at_end_in_bf16="$BF16_LAYERS_AT_END"
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
  +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune="$FLASHINFER_AUTOTUNE"
)

FORWARD_ONLY=(
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN"
)

REWARD=(
  reward_model.reward_manager=dapo
  reward.custom_reward_function.path="$WORKING_DIR/examples/real_nvfp4/math_reward.py"
  reward.custom_reward_function.name=compute_score
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable="$OVERLONG_PENALTY"
  +reward_model.reward_kwargs.overlong_buffer_cfg.len="$OVERLONG_BUFFER_LEN"
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor="$OVERLONG_PENALTY_FACTOR"
  +reward_model.reward_kwargs.overlong_buffer_cfg.log=True
  +reward_model.reward_kwargs.max_resp_len="$MAX_RESPONSE_LENGTH"
)

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
  trainer.use_v1=False
)
if [[ "$RESUME_MODE" = resume_path ]]; then
  TRAINER+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi
if [[ -n "$VERL_WANDB_RUN_ID" ]]; then
  TRAINER+=(
    +trainer.wandb_run_id="$VERL_WANDB_RUN_ID"
    +trainer.wandb_resume="$VERL_WANDB_RESUME"
  )
fi

PRECISION=(actor_rollout_ref.actor.megatron.real_nvfp4.enable=False)
if [[ "$PRECISION_MODE" = real_nvfp4 ]]; then
  [[ -f "$TE_PRECISION_CONFIG" ]]
  PRECISION=(
    actor_rollout_ref.actor.megatron.real_nvfp4.enable=True
    actor_rollout_ref.actor.megatron.real_nvfp4.fp4_format=e2m1
    actor_rollout_ref.actor.megatron.real_nvfp4.fp4_recipe=nvfp4
    actor_rollout_ref.actor.megatron.real_nvfp4.backward_override=dequantized
    actor_rollout_ref.actor.megatron.real_nvfp4.group_size=16
    actor_rollout_ref.actor.megatron.real_nvfp4.fp4_param=False
    actor_rollout_ref.actor.megatron.real_nvfp4.te_precision_config_file="$TE_PRECISION_CONFIG"
    actor_rollout_ref.actor.megatron.real_nvfp4.num_layers_at_start_in_bf16="$BF16_LAYERS_AT_START"
    actor_rollout_ref.actor.megatron.real_nvfp4.num_layers_at_end_in_bf16="$BF16_LAYERS_AT_END"
  )
fi

echo "VERL_REAL_NVFP4_CONTRACT profile=$RUN_PROFILE precision=$PRECISION_MODE scope=all_mlp attention=bf16 rollout_activation=per_token transport=bf16 reload=native r3=1 losses=0of3 bf16_layers=${BF16_LAYERS_AT_START}/${BF16_LAYERS_AT_END} token_mean=1 tis=$ROLLOUT_IS verifier_strict_minerva=$STRICT_MINERVA overlong=${OVERLONG_PENALTY}:${OVERLONG_BUFFER_LEN}:${OVERLONG_PENALTY_FACTOR} autotune=${FLASHINFER_AUTOTUNE} nodes=${NNODES}x${N_GPUS_PER_NODE} tp=1 pp=1 cp=1 ep=4 batch=${TRAIN_PROMPT_BSZ}x${N_RESP_PER_PROMPT} max_num_seqs=$MAX_NUM_SEQS full_adam=1 resume=$RESUME_MODE"

HYDRA_ARGS=(
  --config-path="$WORKING_DIR/recipe/dapo/config" \
  --config-name=dapo_megatron_trainer.yaml \
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
  python3 -m examples.real_nvfp4.main_dapo_compat --cfg job --resolve "${HYDRA_ARGS[@]}"
  exit 0
fi

# Ray jobs do not inherit arbitrary submission-shell variables. Merge the
# validated launcher choice into the supplied runtime env, preserving its other
# fields. STRICT_MINERVA (including explicit 0) takes precedence over YAML.
RUNTIME_ENV_SUBMIT=$(mktemp "${TMPDIR:-/tmp}/verl-real-nvfp4-runtime.XXXXXXXX.json")
trap 'rm -f -- "$RUNTIME_ENV_SUBMIT"' EXIT
python3 - "$RUNTIME_ENV" "$RUNTIME_ENV_SUBMIT" "$STRICT_MINERVA" <<'PY_RUNTIME_ENV'
import json
import sys

import yaml

source, target, strict = sys.argv[1:]
with open(source) as stream:
    runtime_env = yaml.safe_load(stream)
if not isinstance(runtime_env, dict):
    raise ValueError("RUNTIME_ENV must contain a mapping")
env_vars = runtime_env.setdefault("env_vars", {})
if not isinstance(env_vars, dict):
    raise ValueError("RUNTIME_ENV env_vars must contain a mapping")
env_vars["VERL_MATH_DAPO_STRICT_MINERVA"] = strict
with open(target, "w") as stream:
    json.dump(runtime_env, stream)
PY_RUNTIME_ENV
export RAY_ADDRESS
ray job submit --runtime-env="$RUNTIME_ENV_SUBMIT" -- \
  python3 -m examples.real_nvfp4.main_dapo_compat "${HYDRA_ARGS[@]}"
