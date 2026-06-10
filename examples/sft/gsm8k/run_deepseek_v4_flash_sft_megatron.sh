#!/usr/bin/env bash
# DeepSeek-V4-Flash SFT on GSM8K via the Megatron-Bridge backend.
#
# DSv4 hybrid attention requires TP=1; scale with PP and EP -> TP1/PP4/EP8 = 32 GPUs
# (e.g. 4 nodes x 8 GPU on H100/H200, 8 nodes x 4 GPU on GB200/GB300).
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}          # GPUs per node
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-4}
EP_SIZE=${EP_SIZE:-8}
CP_SIZE=${CP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
PAD_MODE=${PAD_MODE:-no_padding}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
LR="5e-6"
MINLR="5e-7"

export VERL_SFT_LOGGING_LEVEL=INFO

backend=${BACKEND:-megatron}

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-4}
RANK=${RANK:-0}

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

# Prepare GSM8K first: python examples/data_preprocess/gsm8k.py
DATASET_DIR=${DATASET_DIR:-$HOME/data/gsm8k}
TRAIN_FILES=${DATASET_DIR}/train.parquet
VAL_FILES=${DATASET_DIR}/test.parquet

project_name=verl_sft_test
RESUME_MODE=disable

MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}
ckpts_home=${ckpts_home:-~/verl/test/gsm8k-sft-deepseek-v4-flash-${backend}}

# DeepSeek-V4-Flash relies on Megatron-Core / Megatron-Bridge support that is not
# in verl's pinned Megatron-Core, so fetch it here and put it on PYTHONPATH (the
# same approach the other megatron SFT examples use to pull feature commits):
#   * DSv4 hybrid attention (CSA + DSA indexer) + packed/THD sequences
#     (engine.use_remove_padding=True) — Megatron-Core PR NVIDIA/Megatron-LM#5011.
#     The SBHD base is already on the `dev` branch; #5011 adds the THD variant.
#   * HF -> Megatron conversion — megatron.bridge (NVIDIA-NeMo/Megatron-Bridge).
# model.mtp.enable=True uses the MTP postprocess kwargs fix in
# verl/models/mcore/mtp_patch.py (included in this PR).
PYPATH=$HOME/pythonpath
mkdir -p "$PYPATH" && cd "$PYPATH"
# Megatron-Core: validated main2dev commit (DSv4 hybrid attention + THD from #5011).
# Pinned because pull/5011/head is rebased over time and can drop main-side symbols
# (e.g. megatron.core._rank_utils.safe_get_world_size) that megatron.bridge imports.
MCORE_COMMIT=ed6b1f65502aec7f2fe27e14a1245c29e435c2a6
[ -d Megatron-LM ] || { git clone https://github.com/NVIDIA/Megatron-LM && \
    (cd Megatron-LM && git fetch origin "$MCORE_COMMIT" && git checkout "$MCORE_COMMIT"); }
# Megatron-Bridge: provides megatron.bridge (the DSv4 AutoBridge / recipes).
[ -d Megatron-Bridge ] || { git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge && \
    (cd Megatron-Bridge && git fetch origin pull/4131/head:dsv4-sft && git checkout dsv4-sft); }
cd -
export PYTHONPATH=${PYTHONPATH:-}:$PYPATH/Megatron-Bridge/src:$PYPATH/Megatron-LM

# DSv4 kernels: fast_hadamard (CSA/DSA indexer) + cuDNN-frontend (THD).
python -c "from fast_hadamard_transform import hadamard_transform" 2>/dev/null || \
    MAX_JOBS=16 pip install --no-build-isolation --no-deps "git+https://github.com/Dao-AILab/fast-hadamard-transform.git"
python -c "from cudnn import DSA" 2>/dev/null || pip install --no-deps "nvidia-cudnn-frontend>=1.24.0" nvidia-cutlass-dsl

MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=${LR} \
    optim.min_lr=${MINLR} \
    optim.lr_warmup_steps=2 \
    optim.weight_decay=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.expert_tensor_parallel_size=1 \
    engine.context_parallel_size=${CP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=False \
    engine.use_remove_padding=${USE_REMOVE_PADDING} \
    engine.use_dist_checkpointing=False \
    +engine.override_transformer_config.apply_rope_fusion=False \
    +engine.override_transformer_config.use_fused_mhc=False \
    +engine.override_transformer_config.recompute_granularity=full \
    +engine.override_transformer_config.recompute_method=uniform \
    +engine.override_transformer_config.recompute_num_layers=1 \
    "

ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
echo "Using megatron engine"
exp_name=gsm8k-deepseek-v4-flash-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-ep${EP_SIZE}-cp${CP_SIZE}

mkdir -p "${ckpts_home}"

torchrun --nnodes=${NNODES} --node_rank=${RANK} --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=1 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.max_length=1024 \
    data.use_dynamic_bsz=False \
    data.messages_key=prompt \
    data.num_workers=0 \
    model.path=$MODEL_PATH \
    model.use_remove_padding=False \
    model.trust_remote_code=True \
    model.mtp.enable=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=-1 \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE}
