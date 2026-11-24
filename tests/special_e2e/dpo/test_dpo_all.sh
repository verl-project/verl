#!/usr/bin/env bash
# DPO end-to-end matrix on 8 GPUs. A 1-GPU run is the golden result; every run with the same
# loss settings must match its step-1 loss and grad norm (compare_dpo_results.py).
set -xeuo pipefail

rm -rf ~/verl/test/log
mkdir -p ~/verl/test/log
export VERL_FILE_LOGGER_ROOT=~/verl/test/log

echo "golden: full fine-tuning on 1 GPU"
NUM_GPUS=1 VERL_FILE_LOGGER_PATH=~/verl/test/log/golden.jsonl bash tests/special_e2e/dpo/run_dpo.sh

echo "8 GPUs, dynamic batching"
NUM_GPUS=8 bash tests/special_e2e/dpo/run_dpo.sh
echo "8 GPUs, fixed micro-batches"
NUM_GPUS=8 USE_DYNAMIC_BSZ=False bash tests/special_e2e/dpo/run_dpo.sh
echo "8 GPUs, sequence parallel 2"
NUM_GPUS=8 SP_SIZE=2 bash tests/special_e2e/dpo/run_dpo.sh

python3 tests/special_e2e/dpo/compare_dpo_results.py

# Runs with other settings: check they complete, not that they match the golden run.
echo "LoRA (reference = policy with adapter disabled)"
NUM_GPUS=8 LORA_RANK=32 VERL_FILE_LOGGER_PATH=~/verl/test/log/lora.jsonl bash tests/special_e2e/dpo/run_dpo.sh
echo "LoRA, resume from the step-2 checkpoint"
NUM_GPUS=8 LORA_RANK=32 TOTAL_TRAIN_STEP=3 RESUME_MODE=auto VERL_FILE_LOGGER_PATH=~/verl/test/log/lora_resume.jsonl \
    bash tests/special_e2e/dpo/run_dpo.sh
echo "reference-free"
NUM_GPUS=8 REFERENCE_FREE=True VERL_FILE_LOGGER_PATH=~/verl/test/log/reffree.jsonl bash tests/special_e2e/dpo/run_dpo.sh
echo "IPO loss"
NUM_GPUS=8 LOSS_TYPE=ipo VERL_FILE_LOGGER_PATH=~/verl/test/log/ipo.jsonl bash tests/special_e2e/dpo/run_dpo.sh

rm -rf ~/verl/test/log
