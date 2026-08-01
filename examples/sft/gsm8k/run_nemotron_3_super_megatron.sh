#!/usr/bin/env bash
# NVIDIA Nemotron 3 Super 120B-A12B SFT with Megatron-Bridge and native MTP.
#
# This launcher is sized for 4 nodes with 8 GPUs per node (32 GPUs total).
# Run it on every node with the same MASTER_ADDR/MASTER_PORT and a distinct
# NODE_RANK in [0, 3], for example:
#   MASTER_ADDR=<node-0-hostname> NODE_RANK=0 bash examples/sft/gsm8k/run_nemotron_3_super_megatron.sh
#
# Dependency snapshot verified against recent upstream main (2026-07-29):
#   Base container: nvcr.io/nvidia/pytorch:26.06-py3 (CUDA 13.3, Python 3.12)
#   Megatron-Bridge: 1f12931e2f34ec26f578a4cffe15adc06f71a5a2
#   Megatron Core: 0.19.0 at cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54
#   Transformer Engine: e7c550c5f80636cf841a8204b1d6f85a5f3f28b7 (2.18.0)
#   Transformers: 5.10.4 (within both verl and Megatron-Bridge constraints)
#
# Keep Megatron Core at the commit vendored by Megatron-Bridge rather than
# installing an unrelated MCore main commit. Build Transformer Engine,
# mamba-ssm, and causal-conv1d against the container's installed Torch/CUDA
# (without build isolation). The current stable verl image's Transformers 5.3
# is too old for this Megatron-Bridge snapshot and must be upgraded.
#
# Reproducible image setup:
#   git clone --recurse-submodules https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
#   cd Megatron-Bridge
#   git checkout 1f12931e2f34ec26f578a4cffe15adc06f71a5a2
#   git submodule update --init --recursive
#   test "$(git -C 3rdparty/Megatron-LM rev-parse HEAD)" = \
#     cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54
#   docker build -f docker/Dockerfile.ci -t verl-nemotron-3-super .
#
# Mount this verl checkout into that image, then install the verl overlay with
# the dependency version shared by both projects:
#   pip install -e /workspace/verl "transformers==5.10.4"

set -xeuo pipefail

# ============================================================
# Distributed: 4 nodes x 8 GPUs
# ============================================================
NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
PYTHON_BIN=${PYTHON_BIN:-python}

# ============================================================
# Data and model
# ============================================================
DATASET_DIR=${DATASET_DIR:-${HOME}/data/gsm8k}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATASET_DIR}/eval.parquet}

MODEL_PATH=${MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}
TOKENIZER_PATH=${TOKENIZER_PATH:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}

# ============================================================
# Parallelism and training
# ============================================================
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-16}
ETP_SIZE=${ETP_SIZE:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-${MAX_LENGTH}}
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-5e-7}
DTYPE=${DTYPE:-bfloat16}

BACKEND=${BACKEND:-megatron}
RESUME_MODE=${RESUME_MODE:-auto}
PROJECT_NAME=${PROJECT_NAME:-verl_sft_gsm8k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-nemotron-3-super-${BACKEND}-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-ep${EP_SIZE}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${HOME}/verl/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${CHECKPOINT_DIR}"

# The official Super recipe repeats its single shared physical MTP block twice.
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-2}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.3}
MTP_DETACH_ENCODER=${MTP_DETACH_ENCODER:-True}

MEGATRON_ENGINE_CONFIG=(
    "engine=${BACKEND}"
    "optim=${BACKEND}"
    "optim.lr=${LR}"
    "optim.min_lr=${MIN_LR}"
    "optim.lr_warmup_steps=10"
    "optim.weight_decay=0.1"
    "optim.betas=[0.9,0.95]"
    "optim.clip_grad=1.0"
    "optim.lr_warmup_init=0"
    "optim.lr_decay_style=cosine"
    "+optim.override_optimizer_config.optimizer_offload_fraction=1"
    "+optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True"
    "+optim.override_optimizer_config.use_precision_aware_optimizer=True"
    "+optim.override_optimizer_config.optimizer_cpu_offload=True"
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
    "+engine.override_ddp_config.data_parallel_sharding_strategy=optim_grads_params"
    "+engine.override_ddp_config.overlap_param_gather=True"
    "+engine.override_ddp_config.overlap_grad_reduce=False"
    "+engine.override_ddp_config.grad_reduce_in_fp32=False"
    "+engine.override_ddp_config.check_for_nan_in_grad=True"
    "engine.override_transformer_config.attention_backend=fused"
    "engine.override_transformer_config.recompute_granularity=full"
    "engine.override_transformer_config.recompute_modules=[core_attn]"
    "engine.override_transformer_config.recompute_method=uniform"
    "engine.override_transformer_config.recompute_num_layers=1"
    "+engine.override_transformer_config.apply_rope_fusion=False"
    "+engine.override_transformer_config.gradient_accumulation_fusion=True"
    "+engine.override_transformer_config.init_method_std=0.014"
    "+engine.override_transformer_config.use_fused_weighted_squared_relu=True"
    "+engine.override_transformer_config.calculate_per_token_loss=True"
    "+engine.override_transformer_config.use_te_rng_tracker=True"
    "+engine.override_transformer_config.moe_token_dispatcher_type=alltoall"
    "+engine.override_transformer_config.moe_shared_expert_overlap=False"
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
    "+engine.override_transformer_config.mtp_use_repeated_layer=True"
    "+engine.override_transformer_config.keep_mtp_spec_in_bf16=True"
)

# CUDA graphs stay disabled above: packed-sequence SFT supplies explicit masks
# that cannot be safely captured/replayed by the model's Mamba layers.
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
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
    "model.mtp.detach_encoder=${MTP_DETACH_ENCODER}" \
    "model.mtp.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR}" \
    "${MEGATRON_ENGINE_CONFIG[@]}" \
    "trainer.test_freq=after_each_epoch" \
    "trainer.save_freq=200" \
    'trainer.logger=["console","wandb"]' \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    "trainer.total_epochs=1" \
    "trainer.default_local_dir=${CHECKPOINT_DIR}" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "trainer.max_ckpt_to_keep=10" \
    "checkpoint.save_contents=[model,optimizer,extra]" \
    "$@"
