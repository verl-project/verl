#!/usr/bin/env bash
set -xeuo pipefail


NUM_GPUS=${NUM_GPUS:-8}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-"fsdp"}  # fsdp or megatron

# Download model if not exists
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=512
max_response_length=1024
enable_overlong_buffer=True
overlong_buffer_len=128
overlong_penalty_factor=1.0

# Training parameters
loss_agg_mode="token-mean"

# Temperature parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

n_gpus_training=8
train_prompt_bsz=128
val_prompt_bsz=128
n_resp_per_prompt=5
train_prompt_mini_bsz=32
test_freq=-1

log_dir="./logs"
mkdir -p $log_dir
timestamp=$(date +"%Y%m%d%H%M%S")
log_file="${log_dir}/qwen2_5-0_5b_transferqueue_${timestamp}.log"

exp_name="$(basename "${MODEL_ID,}")-transferqueue-${ACTOR_STRATEGY}-minimal"

echo "Running transferqueue with ${ACTOR_STRATEGY} strategy"
echo "Total GPUs: ${NUM_GPUS}"

# Common parameters for both FSDP and Megatron
common_params=(
    data.train_files="${HOME}/data/gsm8k/train.parquet"
    data.val_files="${HOME}/data/gsm8k/test.parquet"
    data.prompt_key=prompt
    data.truncation='error'
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts_workers=128
    data.filter_overlong_prompts=True
    data.train_batch_size=${train_prompt_bsz}
    data.val_batch_size=${val_prompt_bsz}
    data.return_raw_chat=${return_raw_chat}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.max_num_batched_tokens=10240
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.mode=${rollout_mode}
    actor_rollout_ref.rollout.disable_log_stats=True
    trainer.logger=console
    trainer.project_name='verl-test-transferqueue'
    trainer.experiment_name="${exp_name}"
    trainer.test_freq="${test_freq}"
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${n_gpus_training}
    trainer.total_training_steps=2
    trainer.total_epochs=15
    trainer.val_before_train=True
)

    # Detect device
    device_name=$(python3 - <<'EOF'
from verl.utils.device import get_device_name
print(get_device_name())
EOF
)

if [ "${ACTOR_STRATEGY}" == "fsdp" ]; then
    echo "Running TransferQueue training with FSDP strategy..."
    # FSDP specific parameters; fsdp_size need to be -1
    gen_tp=1
    sp_size=1
    fsdp_size=-1
    ref_offload=True
    actor_offload=False

    python3 -m verl.experimental.transfer_queue.main_ppo \
        --config-path=config \
        --config-name='transfer_queue_ppo_trainer' \
        "${common_params[@]}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.strategy=fsdp \
        critic.strategy=fsdp \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
        2>&1 | tee "$log_file" $@

elif [ "${ACTOR_STRATEGY}" == "megatron" ]; then
    echo "Running TransferQueue training with Megatron strategy..."
    # Megatron specific parameters
    gen_tp=2
    train_tp=1
    train_pp=2
    ref_offload=True
    actor_offload=False

    extra_flash_args=()

    if [ "$device_name" == "npu" ]; then
        echo "Detect NPU device, enabling FlashAttention..."
        extra_flash_args+=(
            ++actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
        )
    fi

    # For Ascend NPU, please add:
    #++actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    #++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
    python3 -m verl.experimental.transfer_queue.main_ppo \
        --config-path=config \
        --config-name='transfer_queue_ppo_megatron_trainer' \
        "${common_params[@]}" \
        actor_rollout_ref.actor.strategy=megatron \
        critic.strategy=megatron \
        actor_rollout_ref.actor.optim.lr_decay_steps=10000000 \
        actor_rollout_ref.actor.megatron.param_offload=${actor_offload} \
        actor_rollout_ref.actor.megatron.optimizer_offload=${actor_offload} \
        actor_rollout_ref.actor.megatron.grad_offload=${actor_offload} \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
        actor_rollout_ref.ref.megatron.param_offload=${ref_offload} \
        "${extra_flash_args[@]}" \
        2>&1 | tee "$log_file" $@
else
    echo "Error: Unknown strategy ${ACTOR_STRATEGY}. Please use 'fsdp' or 'megatron'"
    exit 1
fi

echo "TransferQueue test completed successfully with ${ACTOR_STRATEGY} strategy"