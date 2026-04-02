#!/usr/bin/env bash
# Legacy Megatron loss normalization — CP>1 smoke test.
#
# Validates that batch_num_tokens is all-reduced over DP*CP (not just DP).
# Default: 1 node x 4 GPUs (PP=2, CP=2)
#
# Usage (single-node, 4 GPUs):
#   bash tests/workers/actor/run_legacy_megatron_loss_norm_multinode_cp.sh
#
# Usage (multi-node, 4 nodes x 1 GPU):
#   NNODES=4 NPROC_PER_NODE=1 ACTOR_PP=2 ACTOR_CP=2 \
#       bash tests/workers/actor/run_legacy_megatron_loss_norm_multinode_cp.sh
#
# Environment variables you may override:
#   NNODES           — number of nodes          (default: 1)
#   NPROC_PER_NODE   — GPUs per node            (default: 4)
#   ACTOR_PP         — pipeline parallelism     (default: 2)
#   ACTOR_TP         — tensor parallelism       (default: 1)
#   ACTOR_CP         — context parallelism      (default: 2)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
ACTOR_PP="${ACTOR_PP:-2}"
ACTOR_TP="${ACTOR_TP:-1}"
ACTOR_CP="${ACTOR_CP:-2}"

echo "=== Legacy Megatron CP smoke: ${NNODES} nodes x ${NPROC_PER_NODE} GPU, PP=${ACTOR_PP} TP=${ACTOR_TP} CP=${ACTOR_CP} ==="

export ACTOR_PP ACTOR_TP ACTOR_CP

exec torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv-backend=c10d \
    "${SCRIPT_DIR}/test_legacy_megatron_loss_normalization.py"
