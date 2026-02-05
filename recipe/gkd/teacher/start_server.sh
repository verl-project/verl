#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

export CUDA_VISIBLE_DEVICES=6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID

BACKEND=vllm
# CKPT_PATH="Qwen/Qwen3-8B"
CKPT_PATH="Qwen/Qwen2.5-14B-Instruct"

ENABLE_PREFIX_CACHING=true
GPU_MEMORY_UTIL=0.75
# MAX_BATCHED_TOKENS=8192 

PROXY_LOG="$SCRIPT_DIR/proxy.log"
WORKER_LOG="$SCRIPT_DIR/worker.log"

wait_server_ready() {
  local server="$1"
  local ip="$2"
  local port="$3"

  while true; do
    echo "wait $server server ready at $ip:$port..."
    local result
    result="$(echo -e "\n" | telnet "$ip" "$port" 2>/dev/null | grep -c "Connected" || true)"
    if [ "$result" -ge 1 ]; then
      break
    fi
    sleep 1
  done
}

wait_for_worker_started() {
  local log_file="$1"
  local pattern="worker started..."

  echo "waiting for teacher vLLM worker to be ready (last line == \"$pattern\")..."

  while [ ! -f "$log_file" ]; do
    sleep 1
  done

  while true; do
    local last_line
    last_line="$(tail -n 1 "$log_file" 2>/dev/null || true)"
    if printf '%s' "$last_line" | grep -Fq "$pattern"; then
      echo "teacher inference server is on..."
      break
    fi
    sleep 1
  done
}

kill_if_running() {
  local pattern="$1"
  pkill -f "$pattern" 2>/dev/null || true
}

# Kill previous instances
kill_if_running "python proxy.py"
kill_if_running "python worker.py"

rm -f "$PROXY_LOG" "$WORKER_LOG"

# Start proxy
nohup python proxy.py >"$PROXY_LOG" 2>&1 &
echo "start teacher proxy (log: $PROXY_LOG)"

wait_server_ready proxy localhost "$PROXY_BACKEND_PORT"
echo "teacher proxy is ready"

# Build worker command
# For Qwen3 Family, need to tune n-logprobs
WORKER_CMD="python worker.py \
  --backend $BACKEND \
  --tp-size 2 \
  --n-logprobs 64 \
  --ckpt-path $CKPT_PATH \
  --gpu-memory-utilization $GPU_MEMORY_UTIL"

# Add optional flags
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
  WORKER_CMD="$WORKER_CMD --enable-prefix-caching"
fi

if [ -n "${MAX_BATCHED_TOKENS:-}" ]; then
  WORKER_CMD="$WORKER_CMD --max-num-batched-tokens $MAX_BATCHED_TOKENS"
fi

# Start worker
nohup $WORKER_CMD >"$WORKER_LOG" 2>&1 &
echo "start teacher worker (log: $WORKER_LOG)"
echo "  - prefix_caching: $ENABLE_PREFIX_CACHING"
echo "  - gpu_memory_util: $GPU_MEMORY_UTIL"
if [ -n "${MAX_BATCHED_TOKENS:-}" ]; then
  echo "  - max_batched_tokens: $MAX_BATCHED_TOKENS"
else
  echo "  - max_batched_tokens: not set (use default)"
fi

wait_for_worker_started "$WORKER_LOG"