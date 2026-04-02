#!/usr/bin/env bash
# Legacy Megatron loss normalization — multi-node PP smoke test.
#
# Profile: PP=ACTOR_PP across multiple nodes.
# Default: 4 nodes x 1 GPU (PP=4, TP=1, CP=1)
#
# Usage (single-node, 2 GPUs):
#   NNODES=1 NPROC_PER_NODE=2 ACTOR_PP=2 \
#       bash tests/workers/actor/run_legacy_megatron_loss_norm_multinode_pp.sh
#
# Usage (multi-node, 4 nodes):
#   bash tests/workers/actor/run_legacy_megatron_loss_norm_multinode_pp.sh
#
# Environment variables you may override:
#   NNODES           — number of nodes          (default: 1)
#   NPROC_PER_NODE   — GPUs per node            (default: 2)
#   ACTOR_PP         — pipeline parallelism     (default: NPROC_PER_NODE * NNODES)
#   ACTOR_TP         — tensor parallelism       (default: 1)
#   ACTOR_CP         — context parallelism      (default: 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
ACTOR_PP="${ACTOR_PP:-${WORLD_SIZE}}"
ACTOR_TP="${ACTOR_TP:-1}"
ACTOR_CP="${ACTOR_CP:-1}"

echo "=== Legacy Megatron PP smoke: ${NNODES} nodes x ${NPROC_PER_NODE} GPU, PP=${ACTOR_PP} TP=${ACTOR_TP} CP=${ACTOR_CP} ==="

export ACTOR_PP ACTOR_TP ACTOR_CP

exec torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv-backend=c10d \
    "${SCRIPT_DIR}/test_legacy_megatron_loss_normalization.py"
