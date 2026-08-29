#!/usr/bin/env bash
# Standalone rollout-only generation on Intel GPU — mirrors examples/generation/run_deepseek_llm_7b.sh.
#
# This path calls verl.trainer.main_generation_server, which ALWAYS calls
# RolloutReplica.init_standalone() (no trainer worker_group involved), unlike
# run_grpo_intel_gpu.sh / run_ppo_intel_gpu.sh which go through init_hybrid().
# Use this to exercise the init_standalone()/init_colocated() device_name path
# in verl/workers/rollout/replica.py on XPU.
#
# Usage:
#   NUM_GPUS=2 bash tests/special_intel_gpu/run_standalone_gen_intel_gpu.sh

set -x

NUM_GPUS=${NUM_GPUS:-2}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}
DATA_PATH=${DATA_PATH:-$HOME/data/gsm8k/test.parquet}
OUTPUT_PATH=${OUTPUT_PATH:-$HOME/data/gsm8k/qwen2_5_05b_intel_gpu_gen_test.parquet}

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    data.train_files="${DATA_PATH}" \
    data.prompt_key=prompt \
    +data.output_path="${OUTPUT_PATH}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.response_length=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${NUM_GPUS} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.n=1 $@
