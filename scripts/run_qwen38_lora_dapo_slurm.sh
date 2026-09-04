#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?run inside a two-node Slurm allocation}"
[[ "${SLURM_PROCID:-0}" == 0 ]] || exit 0
: "${IMAGE:?set the immutable Enroot image}"
: "${TRAIN_MODEL_HEAD:?set the h200-0 BF16 trainer checkpoint}"
: "${TRAIN_MODEL_WORKER:?set the h200-1 BF16 trainer checkpoint}"
: "${ROLLOUT_MODEL:?set the official FP8 checkpoint on both nodes}"
: "${DATA:?set the DAPO-Math dataset directory on both nodes}"
: "${RUN_DIR:?set the node-local result directory}"

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAY_PORT=${RAY_PORT:-6379}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
NETWORK_INTERFACE=${NETWORK_INTERFACE:-ens7}
LOCAL_STORAGE=${LOCAL_STORAGE:-/mnt/ephemeral/$USER}
TRITON_CACHE_ROOT=${TRITON_CACHE_ROOT:-$LOCAL_STORAGE/qwen38/cache/triton}
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-0.98}
ROLLOUT_CACHE_HOST=${ROLLOUT_CACHE_HOST:-}
ROLLOUT_CACHE_READONLY=${ROLLOUT_CACHE_READONLY:-false}
PYTHON_OVERLAY_HOST=${PYTHON_OVERLAY_HOST:-}

[[ "$RAY_NUM_CPUS" =~ ^[1-9][0-9]*$ ]] || {
    echo "RAY_NUM_CPUS must be a positive integer" >&2
    exit 2
}

allocation_nodes=$(scontrol show job -o "$SLURM_JOB_ID" |
    sed -n 's/.* NodeList=\([^ ]*\).*/\1/p')
mapfile -t NODES < <(scontrol show hostnames "$allocation_nodes")
if ((${#NODES[@]} != 2)); then
    echo "expected exactly two Slurm nodes, got ${#NODES[@]}" >&2
    exit 2
fi
HEAD_NODE=${NODES[0]}
WORKER_NODE=${NODES[1]}

if [[ -n "$ROLLOUT_CACHE_HOST" ]]; then
    [[ "$ROLLOUT_CACHE_HOST" == /* ]] || {
        echo "ROLLOUT_CACHE_HOST must be an absolute path" >&2
        exit 2
    }
    if [[ "$ROLLOUT_CACHE_READONLY" == true ]]; then
        rollout_cache_action=repeat
        [[ -d "$ROLLOUT_CACHE_HOST" ]] || {
            echo "rollout cache does not exist: $ROLLOUT_CACHE_HOST" >&2
            exit 2
        }
        [[ -f "$ROLLOUT_CACHE_HOST/SHA256SUMS" ]] || {
            echo "rollout cache is missing SHA256SUMS" >&2
            exit 2
        }
        (cd "$ROLLOUT_CACHE_HOST" && sha256sum --check --quiet SHA256SUMS) || {
            echo "rollout cache failed SHA256 verification" >&2
            exit 2
        }
    else
        rollout_cache_action=cache
        mkdir -p "$ROLLOUT_CACHE_HOST"
    fi
fi
if [[ "${USE_LIGER:-false}" == true && -z "$PYTHON_OVERLAY_HOST" ]]; then
    echo "PYTHON_OVERLAY_HOST is required when USE_LIGER=true" >&2
    exit 2
fi
if [[ -n "$PYTHON_OVERLAY_HOST" && ! -d "$PYTHON_OVERLAY_HOST" ]]; then
    echo "Python overlay does not exist: $PYTHON_OVERLAY_HOST" >&2
    exit 2
fi

node_ipv4() {
    srun --overlap --nodes=1 --ntasks=1 -w "$1" /bin/sh -c \
        "ip -4 -o addr show dev '$NETWORK_INTERFACE' scope global | awk 'NR == 1 {split(\$4, ip, \"/\"); print ip[1]}'"
}

head_ip=$(node_ipv4 "$HEAD_NODE")
[[ -n "$head_ip" ]] || { echo "failed to resolve the Ray head address" >&2; exit 2; }
RAY_ADDRESS=${head_ip}:${RAY_PORT}

export ENROOT_CACHE_PATH="$LOCAL_STORAGE/enroot/cache"
export ENROOT_DATA_PATH="$LOCAL_STORAGE/enroot/data"
export ENROOT_RUNTIME_PATH="$LOCAL_STORAGE/enroot/runtime"
export ENROOT_TEMP_PATH="$LOCAL_STORAGE/enroot/tmp"
mkdir -p "$RUN_DIR"

container=(
    enroot start --rc "$WORKTREE/scripts/enroot_exec.sh"
    -e UV_PROJECT_ENVIRONMENT=/opt/verl-uv-final
    -e UV_CACHE_DIR=/tmp/uv-cache
    -e XDG_CACHE_HOME=/run/xdg-cache
    -e TRITON_CACHE_DIR=/var/tmp
    -e VLLM_DO_NOT_TRACK=1
    -e PYTHONPATH=/workspace
    -e "RAY_memory_usage_threshold=$RAY_MEMORY_USAGE_THRESHOLD"
    -e FLASHINFER_WORKSPACE_BASE=/run/flashinfer
    -e "GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE"
    -e "NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE"
    -m "$WORKTREE:/workspace:none:bind,ro"
)
if [[ -n "$ROLLOUT_CACHE_HOST" ]]; then
    cache_access=rw
    [[ "$ROLLOUT_CACHE_READONLY" == true ]] && cache_access=ro
    container+=(
        -m "$ROLLOUT_CACHE_HOST:/media:none:bind,$cache_access"
        -e ROLLOUT_CACHE_DIR=/media
        -e "ROLLOUT_CACHE_ACTION=$rollout_cache_action"
    )
fi
if [[ -n "$PYTHON_OVERLAY_HOST" ]]; then
    container+=(
        -m "$PYTHON_OVERLAY_HOST:/mnt:none:bind,ro"
        -e PYTHONPATH=/workspace:/mnt
    )
fi

verify_lineage() {
    local node=$1 trainer=$2 output=$3
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
        "${container[@]}" \
        -m "$trainer:/models/q0:none:bind,ro" \
        -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
        "$IMAGE" /opt/verl-uv-final/bin/python \
        /workspace/scripts/verify_fp8_trainer_lineage.py \
        --trainer /models/q0 --rollout /models/qwen38 >"$output"
}

verify_lineage "$HEAD_NODE" "$TRAIN_MODEL_HEAD" "$RUN_DIR/lineage-head.json"
verify_lineage "$WORKER_NODE" "$TRAIN_MODEL_WORKER" "$RUN_DIR/lineage-worker.json"
cmp "$RUN_DIR/lineage-head.json" "$RUN_DIR/lineage-worker.json"

checkpoint_iteration() {
    srun --overlap --nodes=1 --ntasks=1 -w "$1" /bin/sh -c \
        'test ! -f "$1/latest_checkpointed_iteration.txt" || cat "$1/latest_checkpointed_iteration.txt"' \
        sh "$2"
}

copy_checkpoint_metadata() {
    local source_node=$1 source=$2 target_node=$3 target=$4 iteration=$5
    srun --overlap --nodes=1 --ntasks=1 -w "$target_node" \
        test -d "$target/global_step_$iteration/actor"
    srun --overlap --nodes=1 --ntasks=1 -w "$source_node" \
        cat "$source/global_step_$iteration/data.pt" |
        srun --overlap --nodes=1 --ntasks=1 -w "$target_node" /bin/sh -c \
            'cat >"$1.tmp" && mv "$1.tmp" "$1"' sh \
            "$target/global_step_$iteration/data.pt"
    srun --overlap --nodes=1 --ntasks=1 -w "$target_node" /bin/sh -c \
        'printf "%s" "$2" >"$1.tmp" && mv "$1.tmp" "$1"' sh \
        "$target/latest_checkpointed_iteration.txt" "$iteration"
}

sync_checkpoint_metadata() {
    local head_output="$RUN_DIR/ray-head/output"
    local worker_output="$RUN_DIR/ray-worker/output"
    local head_iteration worker_iteration
    head_iteration=$(checkpoint_iteration "$HEAD_NODE" "$head_output")
    worker_iteration=$(checkpoint_iteration "$WORKER_NODE" "$worker_output")
    [[ -z "$head_iteration" || "$head_iteration" =~ ^[0-9]+$ ]] || return 2
    [[ -z "$worker_iteration" || "$worker_iteration" =~ ^[0-9]+$ ]] || return 2
    if [[ -n "$head_iteration" && ${worker_iteration:-0} -lt $head_iteration ]]; then
        copy_checkpoint_metadata \
            "$HEAD_NODE" "$head_output" "$WORKER_NODE" "$worker_output" "$head_iteration"
    elif [[ -n "$worker_iteration" && ${head_iteration:-0} -lt $worker_iteration ]]; then
        copy_checkpoint_metadata \
            "$WORKER_NODE" "$worker_output" "$HEAD_NODE" "$head_output" "$worker_iteration"
    fi
}

sync_checkpoint_metadata

verify_resume_checkpoint() {
    local node=$1 node_run=$2 first_rank=$3 last_rank=$4
    local relative=${RESUME_FROM_PATH#/run/}
    [[ "$relative" != "$RESUME_FROM_PATH" ]] || {
        echo "RESUME_FROM_PATH must be under /run" >&2
        return 2
    }
    srun --overlap --nodes=1 --ntasks=1 -w "$node" /bin/sh -c '
        set -eu
        checkpoint=$1
        first_rank=$2
        last_rank=$3
        test -f "$checkpoint/data.pt"
        rank=$first_rank
        while [ "$rank" -le "$last_rank" ]; do
            for kind in model optim extra_state; do
                test -f "$checkpoint/actor/${kind}_world_size_16_rank_${rank}.pt"
            done
            rank=$((rank + 1))
        done
    ' sh "$node_run/$relative" "$first_rank" "$last_rank"
}

if [[ "${RESUME_MODE:-auto}" == resume_path ]]; then
    verify_resume_checkpoint "$HEAD_NODE" "$RUN_DIR/ray-head" 0 7
    verify_resume_checkpoint "$WORKER_NODE" "$RUN_DIR/ray-worker" 8 15
fi

start_ray() {
    local node=$1 node_run=$2 trainer=$3
    shift 3
    local cache="$TRITON_CACHE_ROOT/$node"
    srun --overlap --nodes=1 --ntasks=1 -w "$node" mkdir -p "$node_run" "$cache"
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
        --output="$node_run/ray.log" --error="$node_run/ray.log" \
        "${container[@]}" \
        -m "$trainer:/models/q0:none:bind,ro" \
        -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
        -m "$DATA:/opt/data:none:bind,ro" \
        -m "$node_run:/run:none:bind,rw" \
        -m "$node_run:/tmp:none:bind,rw" \
        -m "$cache:/var/tmp:none:bind,rw" \
        "$IMAGE" /opt/verl-uv-final/bin/ray start "$@" \
        --temp-dir=/tmp/ray --num-cpus="$RAY_NUM_CPUS" --num-gpus=8 --block &
    RAY_PID=$!
}

stop_ray() {
    local node=$1 node_run=$2 trainer=$3 cache="$TRITON_CACHE_ROOT/$1"
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
        "${container[@]}" \
        -m "$trainer:/models/q0:none:bind,ro" \
        -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
        -m "$DATA:/opt/data:none:bind,ro" \
        -m "$node_run:/run:none:bind,rw" \
        -m "$node_run:/tmp:none:bind,rw" \
        -m "$cache:/var/tmp:none:bind,rw" \
        "$IMAGE" /opt/verl-uv-final/bin/ray stop --force || true
}

head_pid=
worker_pid=
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$worker_pid" ]]; then
        stop_ray "$WORKER_NODE" "$RUN_DIR/ray-worker" "$TRAIN_MODEL_WORKER"
        wait "$worker_pid" 2>/dev/null || true
    fi
    if [[ -n "$head_pid" ]]; then
        stop_ray "$HEAD_NODE" "$RUN_DIR/ray-head" "$TRAIN_MODEL_HEAD"
        wait "$head_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

ray_status() {
    local node_run="$RUN_DIR/ray-head" cache="$TRITON_CACHE_ROOT/$HEAD_NODE"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        "${container[@]}" \
        -m "$TRAIN_MODEL_HEAD:/models/q0:none:bind,ro" \
        -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
        -m "$DATA:/opt/data:none:bind,ro" \
        -m "$node_run:/run:none:bind,rw" \
        -m "$node_run:/tmp:none:bind,rw" \
        -m "$cache:/var/tmp:none:bind,rw" \
        "$IMAGE" /opt/verl-uv-final/bin/ray status --address="$RAY_ADDRESS" \
        >"$RUN_DIR/ray-status.log" 2>&1
}

wait_for_ray() {
    local pattern=$1
    for _ in $(seq 1 30); do
        if ray_status && grep -q "$pattern" "$RUN_DIR/ray-status.log"; then
            return
        fi
        sleep 2
    done
    cat "$RUN_DIR/ray-status.log" >&2
    return 1
}

start_ray "$HEAD_NODE" "$RUN_DIR/ray-head" "$TRAIN_MODEL_HEAD" \
    --head --node-ip-address="$head_ip" --port="$RAY_PORT"
head_pid=$RAY_PID
wait_for_ray Resources
worker_ip=$(node_ipv4 "$WORKER_NODE")
[[ -n "$worker_ip" ]] || { echo "failed to resolve the Ray worker address" >&2; exit 2; }
start_ray "$WORKER_NODE" "$RUN_DIR/ray-worker" "$TRAIN_MODEL_WORKER" \
    --address="$RAY_ADDRESS" --node-ip-address="$worker_ip"
worker_pid=$RAY_PID
wait_for_ray '/16\.0 GPU'

recipe_env=(
    FSDP_SIZE TRAINING_STEPS TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N ROLLOUT_TP
    MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS
    MAX_NUM_SEQS
    LORA_RANK LORA_ALPHA LEARNING_RATE SAVE_FREQ TEST_FREQ RESUME_MODE
    RESUME_FROM_PATH ENABLE_MTP
    ENFORCE_EAGER ROLLOUT_GPU_MEMORY_UTILIZATION USE_REMOVE_PADDING
    USE_DYNAMIC_BSZ PPO_MAX_TOKEN_LEN_PER_GPU LOG_PROB_MAX_TOKEN_LEN_PER_GPU
    MICRO_BATCH_SIZE_PER_GPU ENABLE_GRADIENT_CHECKPOINTING USE_LIGER
    USE_FUSED_KERNELS RESHARD_AFTER_FORWARD VAL_BEFORE_TRAIN
    USE_NO_SYNC_FOR_GRADIENT_ACCUMULATION PAD_TO_LENGTH PAD_TO_LENGTH_BUCKET
    ROLLOUT_CACHE_STEPS EXPECTED_TRAIN_ROWS EXPECTED_VAL_ROWS PROJECT_NAME
    EXPERIMENT_NAME DRY_RUN
)
driver_env=()
for name in "${recipe_env[@]}"; do
    [[ -v $name ]] && driver_env+=(-e "$name=${!name}")
done

node_run="$RUN_DIR/ray-head"
cache="$TRITON_CACHE_ROOT/$HEAD_NODE"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    "${container[@]}" "${driver_env[@]}" \
    -e "RAY_ADDRESS=$RAY_ADDRESS" -e NNODES=2 -e NGPUS_PER_NODE=8 \
    -e TRAIN_MODEL_PATH=/models/q0 -e ROLLOUT_MODEL_PATH=/models/qwen38 \
    -e TRAIN_FILE=/opt/data/train.parquet -e VAL_FILE=/opt/data/val.parquet \
    -e OUTPUT_DIR=/run/output \
    -m "$TRAIN_MODEL_HEAD:/models/q0:none:bind,ro" \
    -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
    -m "$DATA:/opt/data:none:bind,ro" \
    -m "$node_run:/run:none:bind,rw" \
    -m "$node_run:/tmp:none:bind,rw" \
    -m "$cache:/var/tmp:none:bind,rw" \
    "$IMAGE" /bin/bash -lc \
    'source /opt/verl-uv-final/bin/activate; cd /workspace; exec bash scripts/run_qwen38_lora_dapo.sh'
