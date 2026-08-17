#!/usr/bin/env bash
# SFT | NVIDIA Nemotron 3.5 Lightning 30B-A3B | Megatron | NVIDIA H100
#
# This launcher keeps Lightning's native MTP block. Its topology and core
# hyperparameters are adapted from NVIDIA's verified 4K full-SFT recipe:
# 2 nodes x 8 H100, TP=2, EP=8, MBS=1.
# Run it on every node with the same MASTER_ADDR/MASTER_PORT and a distinct
# NODE_RANK, for example:
#   MASTER_ADDR=<node-0-hostname> NODE_RANK=0 \
#     bash examples/sft/gsm8k/run_nemotron_3_5_lightning_megatron.sh
#
# Pinned pre-release software snapshot:
#   Public rebuild base: nvcr.io/nvidia/pytorch:26.06-py3
#   Transformer Engine 2.18.0: e7c550c5f80636cf841a8204b1d6f85a5f3f28b7
#   Megatron-Bridge r0.6.0: c93251151adeeadbae3ff2a2bf5ee7a1c34cff01
#   Megatron-Core 0.19.0 (Bridge submodule): cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54
#   Transformers: >=5.8.1,<5.11 (5.10.4 is compatible with verl and Bridge)
#
# Keep Megatron-Core at the commit vendored by Megatron-Bridge. The public
# Lightning checkpoint revision used for hardware validation is:
#   d468880b6ad3c6e0d21377ce7242adaea4cc884d

set -xeuo pipefail

# ============================================================
# Distributed: 2 nodes x 8 GPUs
# ============================================================
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${NUM_GPUS:-8}}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
PYTHON_BIN=${PYTHON_BIN:-python3}

# ============================================================
# Data and model
# ============================================================
# Prepare the message-formatted SFT dataset with:
#   python examples/data_preprocess/gsm8k_multiturn_sft.py
DATASET_DIR=${DATASET_DIR:-${HOME}/data/gsm8k_sft}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATASET_DIR}/test.parquet}

MODEL_PATH=${MODEL_PATH:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}

# ============================================================
# Parallelism and training
# ============================================================
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
ETP_SIZE=${ETP_SIZE:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-${MAX_LENGTH}}
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-0}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-10}
DTYPE=${DTYPE:-bfloat16}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-after_each_epoch}
RESUME_MODE=${RESUME_MODE:-auto}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-2}
LOGGER=${LOGGER:-'["console","wandb"]'}

BACKEND=${BACKEND:-megatron}
PROJECT_NAME=${PROJECT_NAME:-verl_sft_gsm8k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-nemotron-3-5-lightning-${BACKEND}-tp${TP_SIZE}-ep${EP_SIZE}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${HOME}/verl/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${CHECKPOINT_DIR}"

# HF stores one shared physical Attention+MoE MTP block. The official Bridge
# recipe repeats that block twice during training.
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-2}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.3}
MTP_DETACH_ENCODER=${MTP_DETACH_ENCODER:-False}

MEGATRON_ENGINE_CONFIG=(
    "engine=${BACKEND}"
    "optim=${BACKEND}"
    "optim.lr=${LR}"
    "optim.min_lr=${MIN_LR}"
    "optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
    "optim.weight_decay=0.1"
    "optim.betas=[0.9,0.95]"
    "optim.clip_grad=1.0"
    "optim.lr_warmup_init=0"
    "optim.lr_decay_style=cosine"
    "engine.tensor_model_parallel_size=${TP_SIZE}"
    "engine.pipeline_model_parallel_size=${PP_SIZE}"
    "engine.virtual_pipeline_model_parallel_size=${VPP_SIZE}"
    "engine.context_parallel_size=${CP_SIZE}"
    "engine.expert_model_parallel_size=${EP_SIZE}"
    "engine.expert_tensor_parallel_size=${ETP_SIZE}"
    "engine.sequence_parallel=True"
    "engine.use_mbridge=True"
    "engine.vanilla_mbridge=False"
    "engine.dtype=${DTYPE}"
    "engine.use_remove_padding=True"
    "+engine.override_ddp_config.average_in_collective=False"
    "+engine.override_ddp_config.overlap_grad_reduce=True"
    "+engine.override_ddp_config.overlap_param_gather=False"
    "+engine.override_ddp_config.grad_reduce_in_fp32=False"
    "+engine.override_ddp_config.check_for_nan_in_grad=True"
    "engine.override_transformer_config.attention_backend=fused"
    "engine.override_transformer_config.recompute_granularity=selective"
    "engine.override_transformer_config.recompute_modules=[moe,layernorm,core_attn,mlp]"
    "engine.override_transformer_config.recompute_method=null"
    "engine.override_transformer_config.recompute_num_layers=null"
    "+engine.override_transformer_config.apply_rope_fusion=False"
    "+engine.override_transformer_config.gradient_accumulation_fusion=True"
    "+engine.override_transformer_config.init_method_std=0.0173"
    "+engine.override_transformer_config.use_fused_weighted_squared_relu=True"
    "+engine.override_transformer_config.calculate_per_token_loss=True"
    "+engine.override_transformer_config.use_te_rng_tracker=True"
    "+engine.override_transformer_config.moe_token_dispatcher_type=alltoall"
    "+engine.override_transformer_config.moe_shared_expert_overlap=False"
    "+engine.override_transformer_config.moe_grouped_gemm=True"
    "+engine.override_transformer_config.moe_router_dtype=fp32"
    "+engine.override_transformer_config.moe_router_load_balancing_type=seq_aux_loss"
    "+engine.override_transformer_config.moe_router_bias_update_rate=0.001"
    "+engine.override_transformer_config.moe_permute_fusion=True"
    "+engine.override_transformer_config.moe_enable_deepep=False"
    "+engine.override_transformer_config.moe_aux_loss_coeff=0.0001"
    "+engine.override_transformer_config.moe_router_enable_expert_bias=True"
    "+engine.override_transformer_config.cuda_graph_impl=none"
    "+engine.override_transformer_config.cuda_graph_scope=[]"
    "+engine.override_transformer_config.mtp_num_layers=${MTP_NUM_LAYERS}"
    '+engine.override_transformer_config.mtp_hybrid_override_pattern="*E"'
    "+engine.override_transformer_config.mtp_use_repeated_layer=True"
    "+engine.override_transformer_config.keep_mtp_spec_in_bf16=True"
    "+engine.override_transformer_config.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR}"
)

# CUDA graphs stay disabled above because packed SFT supplies explicit masks to
# the hybrid Mamba model. alltoall is the portable dispatcher default.
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NGPUS_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m verl.trainer.sft_trainer \
    "data.train_files=${TRAIN_FILES}" \
    "data.val_files=${VAL_FILES}" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}" \
    "data.max_length=${MAX_LENGTH}" \
    "data.pad_mode=no_padding" \
    "data.truncation=error" \
    "data.use_dynamic_bsz=True" \
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}" \
    "data.messages_key=messages" \
    "data.ignore_input_ids_mismatch=True" \
    "model.path=${MODEL_PATH}" \
    "model.tokenizer_path=${TOKENIZER_PATH}" \
    "model.use_remove_padding=True" \
    "model.trust_remote_code=True" \
    "model.mtp.enable=True" \
    "model.mtp.enable_train=True" \
    "model.mtp.enable_rollout=False" \
    "model.mtp.detach_encoder=${MTP_DETACH_ENCODER}" \
    "model.mtp.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR}" \
    "${MEGATRON_ENGINE_CONFIG[@]}" \
    "trainer.test_freq=${TEST_FREQ}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.logger=${LOGGER}" \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
    "trainer.default_local_dir=${CHECKPOINT_DIR}" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP}" \
    "checkpoint.save_contents=[model,optimizer,extra]" \
    "$@"
