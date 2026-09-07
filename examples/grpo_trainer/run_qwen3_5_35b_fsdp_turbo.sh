#!/usr/bin/env bash
# dependency: GPU vllm==0.18.0, transformers@<cc7ab9be>
# dependency: NPU vllm==0.18.0, vllm-ascend@<54879467>, transformers@<cc7ab9be>

set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3_5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Qwen3_5-35B}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
NNODES=${NNODES:-1}

GEN_TP=${GEN_TP:-4}
SP_SIZE=${SP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
FSDP_SIZE=${FSDP_SIZE:-}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-35B-A3B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/geo3k/test.parquet"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}
fsdp_size=${FSDP_SIZE:-8}
ep_dispatcher=eager

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        n_devices_per_node=16
        fsdp_size=16
        EP_SIZE=16
        ep_dispatcher=fused
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

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
EXPERT_APPLY_MODULES="['model.language_model.layers.{*}.mlp.experts']"
RECOMPUTE_PLAN="['model.language_model.layers.{*}','model.visual.blocks.{*}']"

ULYSSES_FUNCTION_PATCHES="[{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward'],"\
"type:full_attention},"\
"{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet.forward'],"\
"type:gated_delta_net}]"

LOSS_FUNCTION_PATCHES="[{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],"\
"type:causal_lm_loss}]"

MODULE_PATCHES="[{target:transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeModel.forward,"\
"replacement:fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5_moe_model_forward}]"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=16
    data.max_prompt_length=2048
    data.max_response_length=6144
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
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.strategy=fsdp_turbo
    "${ACTOR_TURBO}.distributed.fully_shard_parallel_size=${fsdp_size}"
    "${ACTOR_TURBO}.distributed.expert_parallel_size=${EP_SIZE}"
    "${ACTOR_TURBO}.distributed.ulysses_parallel_size=${SP_SIZE}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${ACTOR_TURBO}.distributed.ep_plan.apply_modules=${EXPERT_APPLY_MODULES}"
    "+${ACTOR_TURBO}.distributed.ep_plan.dispatcher=${ep_dispatcher}"
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
    "${REF_TURBO}.distributed.fully_shard_parallel_size=${fsdp_size}"
    "${REF_TURBO}.distributed.expert_parallel_size=${EP_SIZE}"
    "${REF_TURBO}.distributed.ulysses_parallel_size=${SP_SIZE}"
    "+${REF_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${REF_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${REF_TURBO}.distributed.ep_plan.apply_modules=${EXPERT_APPLY_MODULES}"
    "+${REF_TURBO}.distributed.ep_plan.dispatcher=${ep_dispatcher}"
    "+${REF_TURBO}.memory.recompute=True"
    "+${REF_TURBO}.memory.recompute_plan=${RECOMPUTE_PLAN}"
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

########################### conditional cp_plan & module_patches ###########################
if [ "${SP_SIZE}" -gt 1 ]; then
    ACTOR+=(
        "+${ACTOR_TURBO}.distributed.cp_plan.ulysses_function_patches=${ULYSSES_FUNCTION_PATCHES}"
        "+${ACTOR_TURBO}.distributed.cp_plan.loss_function_patches=${LOSS_FUNCTION_PATCHES}"
        "+${ACTOR_TURBO}.module_patches=${MODULE_PATCHES}"
    )
    REF+=(
        "+${REF_TURBO}.distributed.cp_plan.ulysses_function_patches=${ULYSSES_FUNCTION_PATCHES}"
        "+${REF_TURBO}.distributed.cp_plan.loss_function_patches=${LOSS_FUNCTION_PATCHES}"
        "+${REF_TURBO}.module_patches=${MODULE_PATCHES}"
    )
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=5
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
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.resume_from_path=checkpoints/
    trainer.val_before_train=True
    trainer.save_freq=-1
    trainer.test_freq=10
    trainer.total_epochs=100
)

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        ROLLOUT+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
            actor_rollout_ref.rollout.max_num_batched_tokens=2048
        )
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee logs/qwen3_5-35b-${start_time}.log
