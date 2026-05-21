#!/usr/bin/env bash
# SFT | GSM8K | FSDP engine | Ascend A3 NPU (16 NPUs per node)

# Examples:
#   # plain SFT on Ascend A3 910C
#   bash run_qwen3_0_6b_ascend_a3_fsdp.sh

set -x

# ---- Ascend A3 910C fixed config ----
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
nproc_per_node=16
master_port=$(shuf -i 20000-65535 -n 1)
# ---- end Ascend A3 910C fixed config ----


# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
SAVE_PATH=${SAVE_PATH:-verl_output}
SP_SIZE=${SP_SIZE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
LR=${LR:-3e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
MAX_LENGTH=${MAX_LENGTH:-4096}
PROJECT_NAME=${PROJECT_NAME:-gsm8k-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gsm8k-sft-qwen3-0-6b-instruct-a3}
# ---- end user-adjustable ----


torchrun --nnodes=1 --nproc_per_node=${nproc_per_node} --master_port=${master_port} \
  -m verl.trainer.sft_trainer \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.truncation=right \
  data.max_length=${MAX_LENGTH} \
  data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
  data.ignore_input_ids_mismatch=True \
  optim.lr=${LR} \
  model.path="${MODEL_PATH}" \
  model.trust_remote_code=True \
  model.use_remove_padding=True \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.default_local_dir="${SAVE_PATH}" \
  trainer.logger='["console"]' \
  data.use_dynamic_bsz=False \
  engine.ulysses_sequence_parallel_size=${SP_SIZE}
