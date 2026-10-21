#!/usr/bin/env bash
# SFT | Moonlight-16B-A3B-Instruct | Automodel (nemo_automodel) engine | 8 GPU
# Set USE_LORA=1 to enable Automodel's built-in PEFT support.
# Based on examples/sft/gsm8k/run_qwen3_30b_automodel.sh (FSDP2, EP8, TE backends) but
# targets Moonlight-16B-A3B-Instruct (DeepseekV2 MoE arch).

set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-moonshotai/Moonlight-16B-A3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$HOME/dataset/hellaswag_sft/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/dataset/hellaswag_sft/validation.parquet}
USE_LORA=${USE_LORA:-0}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGETS=${LORA_TARGETS:-all-linear}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}

WEIGHT_DECAY=${WEIGHT_DECAY:-0}

SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-10}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}

lora_args=()
case "${USE_LORA}" in
    0)
        LR=${LR:-1e-5}
        PROJECT_NAME=${PROJECT_NAME:-automodel-moonlight16b-sft}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-moonlight16b_full}
        ;;
    1)
        LR=${LR:-1e-4}
        PROJECT_NAME=${PROJECT_NAME:-automodel-moonlight16b-sft-lora}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-moonlight16b_lora}
        lora_args+=(
            "model.lora_rank=${LORA_RANK}"
            "model.lora_alpha=${LORA_ALPHA}"
            "model.target_modules=${LORA_TARGETS}"
        )
        ;;
    *)
        echo "USE_LORA must be 0 or 1, got '${USE_LORA}'" >&2
        exit 2
        ;;
esac

# TE 2.17 prefers cuDNN FusedAttention over FlashAttention on Hopper "for performance",
# but cuDNN 9.20's fused_attn_f16_arbitrary_seqlen thd backward hits CUDNN_STATUS_BAD_PARAM
# in the reshape op. Force FlashAttention so the thd (remove-padding) backward works.
export NVTE_FUSED_ATTN=0
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

torchrun --standalone --nnodes=${NNODES} --nproc_per_node=${NDEVICES_PER_NODE} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.truncation=left \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    data.train_max_samples=-1 \
    data.val_max_samples=1024 \
    \
    model=hf_model \
    model.path="${MODEL_PATH}" \
    model.trust_remote_code=True \
    model.use_remove_padding=True \
    "${lora_args[@]}" \
    \
    engine=automodel \
    engine.distributed_strategy=fsdp2 \
    engine.tp_size=1 \
    engine.pp_size=1 \
    engine.cp_size=1 \
    engine.ep_size=8 \
    engine.backend_config.dispatcher=deepep \
    engine.backend_config.attn=te \
    engine.backend_config.linear=te \
    engine.backend_config.rms_norm=torch_fp32 \
    engine.backend_config.enable_fsdp_optimizations=True \
    engine.backend_config.experts=torch_mm \
    engine.activation_checkpointing=True \
    engine.model_dtype=bf16 \
    engine.attn_implementation=te \
    engine.use_torch_compile=False \
    \
    optim=automodel \
    optim.optimizer=FusedAdam \
    optim.optimizer_impl=transformer_engine.pytorch.optimizers.fused_adam \
    optim.lr=${LR} \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=${WEIGHT_DECAY} \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.init_lr_ratio=0.1 \
    optim.min_lr_ratio=0.01 \
    optim.lr_scheduler_type=cosine \
    optim.master_weights=true \
    optim.store_param_remainders=true \
    optim.exp_avg_dtype=bf16 \
    optim.exp_avg_sq_dtype=bf16 \
    \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.logger=console \
    trainer.seed=1111 \
    trainer.nnodes=${NNODES} \
    trainer.resume_mode=disable \
    "$@"
