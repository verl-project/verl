#!/usr/bin/env bash
# GRPO | DeepSeek-V4-Flash | vLLM rollout | FSDPTurbo training | Ascend NPU
# with FSDPTurbo: https://gitcode.com/Ascend/FSDPTurbo

set -xeuo pipefail

INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-GRPO-DeepSeekV4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV4-Flash}

RAY_DATA_HOME="/verl"
MODEL_PATH="/models/DeepSeek-V4-Flash-BF16"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
TRAIN_FILE="/datasets/dapo-math-17k.parquet"
TEST_FILE="/datasets/dapo-math-17k.parquet"

########################### derived defaults ###########################
NNODES=${NNODES:-4}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-16}

GEN_TP=${GEN_TP:-4}
GEN_DP=${GEN_DP:-8}
GEN_EP=${GEN_EP:-32}
SP_SIZE=${SP_SIZE:-1}
EP_SIZE=${EP_SIZE:-32}
TP_SIZE=${TP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-64}

ep_dispatcher=fused
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 6))
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

: "${FSDPTURBO_ROOT:?Set FSDPTURBO_ROOT to the FSDPTurbo checkout path available on every Ray node}"
FSDPTURBO_ROOT=$(cd -- "${FSDPTURBO_ROOT}" && pwd -P)
if [[ ! -f "${FSDPTURBO_ROOT}/fsdp_turbo/fsdp_turbo.py" ||
      ! -f "${FSDPTURBO_ROOT}/examples/deepseekv4/modeling_deepseek_v4.py" ]]; then
    printf 'Invalid FSDPTurbo checkout: %s\n' "${FSDPTURBO_ROOT}" >&2
    exit 1
fi
export PYTHONPATH="${FSDPTURBO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

########################### shared turbo config values ###########################
ACTOR_TURBO="actor_rollout_ref.actor.fsdp_config.turbo_config"
REF_TURBO="actor_rollout_ref.ref.fsdp_config.turbo_config"

# DeepSeekV4 is a pure-text model: no model.visual.* blocks.
FSDP_APPLY_MODULES='{model.embed_tokens:{},'\
'model.layers.\{*\}:{},'\
'lm_head:{}}'

HOOK_MODULES="['model.layers.{*}']"
EXPERT_APPLY_MODULES="['model.layers.{*}.mlp.experts']"
RECOMPUTE_PLAN="['model.layers.{*}']"

# DeepSeekV4 native operators swapped for FSDPTurbo fused implementations.
MODULE_PATCHES="[{target:transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4RMSNorm,"\
"replacement:examples.deepseekv4.modeling_deepseek_v4.FSDPTurboDeepSeekV4RMSNorm},"\
"{target:transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4HyperConnection,"\
"replacement:examples.deepseekv4.modeling_deepseek_v4.FSDPTurboDeepSeekV4HyperConnection}]"

########################### parameter arrays ###########################
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False

    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=64
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.filter_overlong_prompts_workers=0
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=False
    +actor_rollout_ref.model.override_config.attn_implementation=eager
    +actor_rollout_ref.model.override_config._experts_implementation=eager
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp_turbo

    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.optimizer=AdamW

    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False

    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False

    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=False
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

    "${ACTOR_TURBO}.distributed.fully_shard_parallel_size=${FSDP_SIZE}"
    "${ACTOR_TURBO}.distributed.expert_parallel_size=${EP_SIZE}"
    "${ACTOR_TURBO}.distributed.expert_fully_shard_parallel_size=2"
    "${ACTOR_TURBO}.distributed.tensor_parallel_size=${TP_SIZE}"
    "${ACTOR_TURBO}.distributed.ulysses_parallel_size=${SP_SIZE}"

    "+${ACTOR_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=0"
    "actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=0"
    "actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.reduce_dtype=bf16"

    "+${ACTOR_TURBO}.distributed.ep_plan.apply_modules=${EXPERT_APPLY_MODULES}"
    "+${ACTOR_TURBO}.distributed.ep_plan.dispatcher=${ep_dispatcher}"

    "+${ACTOR_TURBO}.memory.recompute=True"
    "+${ACTOR_TURBO}.memory.recompute_plan=${RECOMPUTE_PLAN}"
    "+${ACTOR_TURBO}.module_patches=${MODULE_PATCHES}"
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp_turbo
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.use_torch_compile=False

    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    "${REF_TURBO}.distributed.fully_shard_parallel_size=${FSDP_SIZE}"
    "${REF_TURBO}.distributed.expert_parallel_size=${EP_SIZE}"
    "${REF_TURBO}.distributed.expert_fully_shard_parallel_size=2"
    "${REF_TURBO}.distributed.tensor_parallel_size=${TP_SIZE}"
    "${REF_TURBO}.distributed.ulysses_parallel_size=${SP_SIZE}"

    "+${REF_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${REF_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=0"
    "actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=0"
    "actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.reduce_dtype=bf16"

    "+${REF_TURBO}.distributed.ep_plan.apply_modules=${EXPERT_APPLY_MODULES}"
    "+${REF_TURBO}.distributed.ep_plan.dispatcher=${ep_dispatcher}"

    "+${REF_TURBO}.module_patches=${MODULE_PATCHES}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.load_format="safetensors"

    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.data_parallel_size=${GEN_DP}
    actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP}

    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1

    actor_rollout_ref.rollout.gpu_memory_utilization=0.56
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False

    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512

    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=48
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
)

TRAINER=(
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.logger=['console']

    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}

    trainer.critic_warmup=0
    trainer.balance_batch=False
    trainer.total_epochs=100
    trainer.total_training_steps=100

    trainer.resume_from_path=checkpoints/
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
)

########################### launch ###########################
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PYTHONPATH}" \
    "$@"
