#!/bin/bash
# 多节点 Ray 入口脚本：根据 NODE_RANK 和 HEAD_ADDRESS 在 head 上起 Ray 并跑训练，在 worker 上起 Ray 并挂到 head。
#
# 环境变量：
#   NODE_RANK          - 当前节点序号，0=head，>=1=worker（默认 0）
#   HEAD_ADDRESS       - head 节点地址，格式 ip:port 或 hostname:port（worker 必填）
#   RAY_HEAD_ADDRESS   - 同上，与 HEAD_ADDRESS 二选一
#   RAY_PORT           - Ray 端口（默认 6379）
#   HEAD_IP / POD_IP   - 可选，head 本机 IP（不设则用 hostname -i 推断）
#   RAY_MIN_GPUS       - 可选，head 上等待集群达到该 GPU 数再跑训练（如 16）
#   RAY_WAIT_TIMEOUT_SEC - 等待 GPU 的超时秒数（默认 120）
#   NUM_GPUS_PER_NODE  - 每节点 GPU 数（默认 8）
#   NUM_CPUS_PER_NODE  - 每节点 CPU 数（默认 96）
#
# 用法示例：
#   # 单节点（仅 head）
#   NODE_RANK=0 ./run_ray_multinode.sh
#
#   # 两节点：head 上（假设本机 IP 由调度器或 POD_IP 提供）
#   NODE_RANK=0 ./run_ray_multinode.sh
#
#   # 两节点：worker 上（HEAD_ADDRESS 需为 head 的 ip:6379，由调度器或 K8s 服务名注入）
#   NODE_RANK=1 HEAD_ADDRESS=<head_ip>:6379 ./run_ray_multinode.sh
#
#   # 指定训练脚本并等待 16 卡再开训
#   NODE_RANK=0 RAY_MIN_GPUS=16 ./run_ray_multinode.sh -- bash -exc /path/to/train_forward_rdkit_grpo_min.sh
#
# 若 rjob 支持按副本传不同 env，可这样用 -P 2：
#   - 副本 0：NODE_RANK=0，执行 run_ray_multinode.sh（脚本内会起 head 并跑训练）
#   - 副本 1：NODE_RANK=1 HEAD_ADDRESS=<head 的 ip 或服务名>:6379，执行 run_ray_multinode.sh（仅起 worker 并 block）
# 其中 HEAD_ADDRESS 若由调度器提供，可能是 JOB_NAME-0.NAMESPACE 或 MASTER_ADDR 等，需与集群约定。

set -e

RAY_PORT=${RAY_PORT:-6379}
NODE_RANK=${NODE_RANK:-0}
# HEAD_ADDRESS 或 RAY_HEAD_ADDRESS，格式 ip:port 或 hostname:port
HEAD_ADDRESS=${HEAD_ADDRESS:-${RAY_HEAD_ADDRESS}}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
NUM_CPUS_PER_NODE=${NUM_CPUS_PER_NODE:-96}
# 可选：head 上等待集群达到目标 GPU 数再跑训练（避免 worker 未连上就开训）
RAY_MIN_GPUS=${RAY_MIN_GPUS:-}
RAY_WAIT_TIMEOUT_SEC=${RAY_WAIT_TIMEOUT_SEC:-120}

# 从 "--" 后或第一个非选项参数取要执行的命令；若没有则用默认训练脚本
TRAIN_CMD=()
if [[ "$1" == "--" ]]; then
  shift
  TRAIN_CMD=("$@")
elif [[ $# -gt 0 && "$1" != --* ]]; then
  TRAIN_CMD=("$@")
fi
if [[ ${#TRAIN_CMD[@]} -eq 0 ]]; then
  TRAIN_CMD=(bash -exc "/mnt/shared-storage-user/mineru4s/dingruiyi/verl_wanjuan/train_forward_rdkit_grpo_min.sh")
fi

ulimit -n 65535
export RAY_DEDUP_LOGS=0

# 获取本机 IP（head 节点用）
get_self_ip() {
  if [[ -n "$HEAD_IP" ]]; then
    echo "$HEAD_IP"
    return
  fi
  if [[ -n "$POD_IP" ]]; then
    echo "$POD_IP"
    return
  fi
  local ip
  ip=$(hostname -i 2>/dev/null | awk '{print $1}')
  if [[ -n "$ip" ]]; then
    echo "$ip"
    return
  fi
  ip=$(hostname -I 2>/dev/null | awk '{print $1}')
  if [[ -n "$ip" ]]; then
    echo "$ip"
    return
  fi
  echo "127.0.0.1"
}

if [[ "$NODE_RANK" -eq 0 ]]; then
  # ---------- Head 节点：启动 Ray head，再执行训练 ----------
  self_ip=$(get_self_ip)
  export RAY_HEAD_ADDRESS="${self_ip}:${RAY_PORT}"
  echo "[NODE_RANK=0] Starting Ray HEAD at $RAY_HEAD_ADDRESS (self_ip=$self_ip)"
  ray start --head \
    --node-ip-address="$self_ip" \
    --port="$RAY_PORT" \
    --num-cpus "${NUM_CPUS_PER_NODE}" \
    --num-gpus "${NUM_GPUS_PER_NODE}" \
    --block &
  RAY_HEAD_PID=$!
  # 给 head 稳定时间，再让 worker 连
  sleep 10
  # 若设置了 RAY_MIN_GPUS，则轮询直到集群 GPU 数达标或超时
  if [[ -n "$RAY_MIN_GPUS" && "$RAY_MIN_GPUS" -gt 0 ]]; then
    echo "[NODE_RANK=0] Waiting for at least $RAY_MIN_GPUS GPUs (timeout ${RAY_WAIT_TIMEOUT_SEC}s)..."
    waited=0
    while [[ $waited -lt $RAY_WAIT_TIMEOUT_SEC ]]; do
      n=$(python3 -c "
import os, sys
os.environ.setdefault('RAY_ADDRESS', 'auto')
try:
  import ray
  ray.init(address='auto', ignore_reinit_error=True)
  r = ray.cluster_resources()
  g = int(r.get('GPU', 0))
  print(g)
except Exception as e:
  print(0, file=sys.stderr)
" 2>/dev/null || echo "0")
      if [[ -n "$n" && "$n" -ge "$RAY_MIN_GPUS" ]]; then
        echo "[NODE_RANK=0] Cluster has $n GPUs, proceeding."
        break
      fi
      sleep 5
      waited=$((waited + 5))
    done
    if [[ $waited -ge $RAY_WAIT_TIMEOUT_SEC ]]; then
      echo "[NODE_RANK=0] WARNING: Timeout waiting for $RAY_MIN_GPUS GPUs. Proceeding anyway."
    fi
  fi
  echo "[NODE_RANK=0] Ray head started. Running training: ${TRAIN_CMD[*]}"
  export RAY_ADDRESS="auto"
  "${TRAIN_CMD[@]}"
  train_exit=$?
  kill $RAY_HEAD_PID 2>/dev/null || true
  exit $train_exit
else
  # ---------- Worker 节点：连到 head 并 block ----------
  if [[ -z "$HEAD_ADDRESS" ]]; then
    echo "ERROR: NODE_RANK=$NODE_RANK but HEAD_ADDRESS (or RAY_HEAD_ADDRESS) is not set." >&2
    exit 1
  fi
  echo "[NODE_RANK=$NODE_RANK] Starting Ray worker, connecting to $HEAD_ADDRESS"
  ray start --address "$HEAD_ADDRESS" \
    --num-cpus "${NUM_CPUS_PER_NODE}" \
    --num-gpus "${NUM_GPUS_PER_NODE}" \
    --block
fi
