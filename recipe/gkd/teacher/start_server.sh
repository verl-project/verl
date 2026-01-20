#!/usr/bin/env bash
set -euo pipefail

# Always run relative to this script's directory (support any CWD launch)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

export CUDA_VISIBLE_DEVICES=6
export CUDA_DEVICE_ORDER=PCI_BUS_ID

BACKEND=vllm
# CKPT_PATH="/path/to/TEACHER_MODEL/"
CKPT_PATH="Qwen/Qwen2.5-14B-Instruct"

PROXY_LOG="$SCRIPT_DIR/proxy.log"
WORKER_LOG="$SCRIPT_DIR/worker.log"

wait_server_ready() {
  local server="$1"
  local ip="$2"
  local port="$3"

  while true; do
    echo "wait $server server ready at $ip:$port..."
    # Telnet prints "Connected to ..." on success.
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

  # Wait for log file to exist
  while [ ! -f "$log_file" ]; do
    sleep 1
  done

  # Poll until the *last line* matches the pattern
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
  # pkill -f is simpler/safer than parsing ps output
  pkill -f "$pattern" 2>/dev/null || true
}

# Kill previous instances (if any)
kill_if_running "python proxy.py"
kill_if_running "python worker.py"

# (Optional) remove old logs so we don't accidentally match a previous run's last line
rm -f "$PROXY_LOG" "$WORKER_LOG"

# Start proxy
nohup python proxy.py >"$PROXY_LOG" 2>&1 &
echo "start teacher proxy (log: $PROXY_LOG)"

# Wait for proxy backend port ready
wait_server_ready proxy localhost "$PROXY_BACKEND_PORT"
echo "teacher proxy is ready"

# Start worker (teacher inference server)
nohup python worker.py \
  --backend "$BACKEND" \
  --tp-size 1 \
  --n-logprobs 256 \
  --ckpt-path "$CKPT_PATH" \
  >"$WORKER_LOG" 2>&1 &
echo "start teacher worker (log: $WORKER_LOG)"

# Wait until worker signals readiness via log last-line
wait_for_worker_started "$WORKER_LOG"