#!/usr/bin/env bash
# Tested with 1 & 4 GPUs
set -xeuo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
OUTPUT_PATH=${OUTPUT_PATH:-$HOME/data/gen/qwen_05_gen_test.parquet}
GEN_TP=${GEN_TP:-2}  # Default tensor parallel size to 2

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    data.train_files="${HOME}/data/gsm8k/test.parquet" \
    data.prompt_key=prompt \
    +data.output_path="${OUTPUT_PATH}" \
    actor_rollout_ref.model.path="${MODEL_ID}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
