#!/usr/bin/env bash
# main_generation_server | text | vLLM rollout | NVIDIA GPUs
# Rollout-only generation with DeepSeek-LLM-7B-Chat.
#
# Single-node (default):
#   bash run_deepseek_llm_7b.sh
#
# Multi-node (e.g. 2 nodes, rollout_tp=16):
#   NNODES=2 ROLLOUT_TP=16 bash run_deepseek_llm_7b.sh

set -xeuo pipefail

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-deepseek-ai/deepseek-llm-7b-chat}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

DATA_PATH=${DATA_PATH:-$HOME/data/gsm8k/test.parquet}
OUTPUT_PATH=${OUTPUT_PATH:-$HOME/data/gsm8k/deepseek_llm_7b_gen_test.parquet}

PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-1024}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
N_SAMPLES=${N_SAMPLES:-1}
# ---- end user-adjustable ----

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    data.train_files="${DATA_PATH}" \
    data.prompt_key=prompt \
    +data.output_path="${OUTPUT_PATH}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n="${N_SAMPLES}" "$@"
