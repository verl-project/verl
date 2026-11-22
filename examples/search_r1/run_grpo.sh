#!/usr/bin/env bash
# Run inside the selected stable image with the project and sd-search mounted.
# Dependencies and parquet must be prepared before this offline worker starts.
set -euo pipefail

VERL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SD_ROOT=${SD_ROOT:-/mnt/shared-storage-gpfs2/gpfs2-agent-multimodel/user/wuyinqi/autoResearch/sd-search}
MODEL_PATH=${MODEL_PATH:-$SD_ROOT/models/Qwen2.5-3B}
DATA_DIR=${DATA_DIR:-$VERL_ROOT/../data/search_r1_sd_search}
VAL_FILE=${VAL_FILE:-$DATA_DIR/monitor.parquet}
RUN_DIR=${RUN_DIR:-$VERL_ROOT/../outputs/search-r1-$(date +%Y%m%d-%H%M%S)}
TRAIN_PYTHON=${TRAIN_PYTHON:-/usr/bin/python3}
RETRIEVER_PYTHON=${RETRIEVER_PYTHON:-$SD_ROOT/.envs/sd-search/bin/python}
START_RETRIEVER=${START_RETRIEVER:-1}
DRY_RUN=${DRY_RUN:-0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
# New verl multiplies this prompt count by ROLLOUT_N: 256 * 5 = 1280 trajectories.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-32}
OLD_LOG_PROB_MICRO_BATCH_SIZE=${OLD_LOG_PROB_MICRO_BATCH_SIZE:-32}
REF_LOG_PROB_MICRO_BATCH_SIZE=${REF_LOG_PROB_MICRO_BATCH_SIZE:-32}
TOTAL_STEPS=${TOTAL_STEPS:-200}
export N_GPUS=${N_GPUS:-4}
export RETRIEVAL_GPU_INDICES=${RETRIEVAL_GPU_INDICES:-0,1,2,3}
ROLLOUT_N=${ROLLOUT_N:-5}
# Total vLLM memory budget; includes weights, activations and the KV cache.
# The four-GPU retriever remains resident on a subset of the training devices.
ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.60}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS:-8192}

mkdir -p "$RUN_DIR/retrieval"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
export RUN_DIR
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false VERL_USE_UV=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export WANDB_MODE=${WANDB_MODE:-offline}
cd "$VERL_ROOT"

if [[ "$DRY_RUN" != 1 ]]; then
"$TRAIN_PYTHON" - <<'PY'
import torch
import json, os, re
from pathlib import Path
from examples.search_r1.gpu_layout import retrieval_gpu_layout
expected = int(os.environ['N_GPUS'])
if expected not in [4, 8] or torch.cuda.device_count() != expected:
    raise RuntimeError(f'This recipe requires exactly {expected} visible training GPUs')
devices = []
for i in range(expected):
    match = re.search(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', str(torch.cuda.get_device_properties(i).uuid))
    devices.append({'uuid': 'GPU-'+match[0] if match else None})
layout = retrieval_gpu_layout(devices, map(int, os.environ['RETRIEVAL_GPU_INDICES'].split(',')))
directory = Path(os.environ['RUN_DIR'])/'retrieval'
(directory/'gpu-device-map.json').write_text(json.dumps(layout, indent=2)+'\n')
(directory/'cuda-visible-devices.txt').write_text(layout['cuda_visible_devices']+'\n')
print('GPU_CHECK_OK', [torch.cuda.get_device_name(i) for i in range(expected)], 'RETRIEVAL_LAYOUT', layout, flush=True)
PY
test -f "$DATA_DIR/train.parquet"
test -f "$VAL_FILE"
test -f "$MODEL_PATH/config.json"
fi

retrieval_pid=
cleanup() {
    if [[ -n "$retrieval_pid" ]]; then
        kill "$retrieval_pid" 2>/dev/null || true
        wait "$retrieval_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$DRY_RUN" != 1 && "$START_RETRIEVER" == 1 ]]; then
    "$TRAIN_PYTHON" - <<'PY'
import socket
with socket.socket() as connection:
    connection.settimeout(1)
    if connection.connect_ex(('127.0.0.1', 8000)) == 0:
        raise RuntimeError('Port 8000 is already in use; use START_RETRIEVER=0 to reuse a verified service')
PY
    read -r retrieval_cuda_mask < "$RUN_DIR/retrieval/cuda-visible-devices.txt"
    env CUDA_VISIBLE_DEVICES="$retrieval_cuda_mask" \
        PYTHONPATH="$SD_ROOT/.envs/faiss-gpu-cu12-1.8.0.2:$SD_ROOT/src" \
        NVIDIA_TF32_OVERRIDE=0 OPENBLAS_NUM_THREADS=20 OMP_NUM_THREADS=20 \
        "$RETRIEVER_PYTHON" -u -m sd_search.retrieval \
        --index "$SD_ROOT/data/retrieval/e5_Flat.index" \
        --corpus "$SD_ROOT/data/retrieval/wiki-18.jsonl" \
        --offsets "$SD_ROOT/data/retrieval/wiki-18.offsets.u64" \
        --model "$SD_ROOT/models/e5-base-v2" \
        --device cpu --threads 20 --search-backend gpu --gpu-scratch-mib 1536 --port 8000 \
        --timing-log "$RUN_DIR/retrieval/timing.jsonl" \
        --startup-report "$RUN_DIR/retrieval/startup.json" \
        > "$RUN_DIR/retrieval/service.log" 2>&1 &
    retrieval_pid=$!
fi
ready=0
if [[ "$DRY_RUN" != 1 ]]; then
for ((i=0; i<300; i++)); do
    if [[ -n "$retrieval_pid" ]] && ! kill -0 "$retrieval_pid" 2>/dev/null; then
        cat "$RUN_DIR/retrieval/service.log"
        exit 1
    fi
    if curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
        http://127.0.0.1:8000/health > "$RUN_DIR/retrieval/health.json"; then
        ready=1
        break
    fi
    sleep 2
done
[[ "$ready" == 1 ]] || { echo "Retriever health check timed out"; exit 1; }
fi

if [[ "$DRY_RUN" != 1 && -n "${ROLLOUT_VERIFICATION_SCRIPT:-}" ]]; then
    "$TRAIN_PYTHON" "$ROLLOUT_VERIFICATION_SCRIPT" retrieval
fi
if [[ "$DRY_RUN" != 1 && -n "${ROLLOUT_DIAGNOSTIC_SCRIPT:-}" ]]; then
    "$TRAIN_PYTHON" "$ROLLOUT_DIAGNOSTIC_SCRIPT" before
fi

COMMAND=("$TRAIN_PYTHON" -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" data.val_batch_size=128 \
    data.max_prompt_length=1024 data.max_response_length=4096 \
    data.filter_overlong_prompts=False data.truncation=error data.seed=42 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attention_dropout=0.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_seqs=1024 actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$OLD_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$VERL_ROOT/examples/search_r1/config/tools.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=search_r1 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$VERL_ROOT/examples/search_r1/config/agent_loop.yaml" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward.custom_reward_function.path="$VERL_ROOT/examples/search_r1/protocol.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 trainer.logger='[console]' \
    trainer.project_name=search-r1 trainer.experiment_name=search-r1-qwen25-3b-multimessage \
    trainer.total_epochs=1 trainer.total_training_steps="$TOTAL_STEPS" \
    +trainer.log_old_policy_entropy=False actor_rollout_ref.actor.calculate_entropy=False \
    trainer.val_before_train=False trainer.save_freq=50 trainer.test_freq=50 \
    trainer.default_local_dir="$RUN_DIR/checkpoints" \
    trainer.rollout_data_dir="$RUN_DIR/rollouts" trainer.validation_data_dir="$RUN_DIR/validation" \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    "$@")
"$TRAIN_PYTHON" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:],indent=2)+"\n")' \
    "$RUN_DIR/command.json" "${COMMAND[@]}"
if [[ "$DRY_RUN" == 1 ]]; then
    cat "$RUN_DIR/command.json"
else
    "${COMMAND[@]}"
    if [[ -n "${ROLLOUT_DIAGNOSTIC_SCRIPT:-}" ]]; then
        "$TRAIN_PYTHON" "$ROLLOUT_DIAGNOSTIC_SCRIPT" after
    fi
fi
