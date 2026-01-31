#!/bin/bash

export HF_HOME="/wekafs/0_public/huggingface"

# MODEL_PATH=ibm-granite/granite-4.0-h-tiny-base
MODEL_PATH=ibm-granite/granite-4.0-h-small

export HF_HOME="/wekafs/0_public/huggingface"

python compare.py \
    --engine vllm \
    --model $MODEL_PATH \
    --max-new-tokens 1024 \
    --vllm-gpu-util 0.8 \
    --dtype float16 \
    --n 1 \
    --out out \
    --prefill-batch-size 1 \
    --batch-size 16 \
    --is-mamba-like \
    2>&1 | tee log.txt

# --num-prompts 16 \
