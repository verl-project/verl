#!/bin/bash

export HF_HOME="/wekafs/0_public/huggingface"

python compare.py \
    --engine vllm \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --max-new-tokens 1024 \
    --vllm-gpu-util 0.8 \
    --dtype float16 \
    --n 1 \
    --out out \
    --profile-trace \
    2>&1 | tee log.txt
