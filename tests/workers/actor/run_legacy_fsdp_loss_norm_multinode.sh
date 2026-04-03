#!/usr/bin/env bash
# Simple distributed smoke script for legacy FSDP loss normalization.
# Can be adapted for multi-node environments (rendezvous details will
# need to be supplied via MASTER_ADDR / MASTER_PORT or similar).
#
# Default: single-node, 2 GPUs.
#
# Usage (single-node, 2 GPUs):
#   bash tests/workers/actor/run_legacy_fsdp_loss_norm_multinode.sh
#
# Usage (single-node, 4 GPUs):
#   NPROC_PER_NODE=4 MINI_BATCH_SIZE=8 \
#       bash tests/workers/actor/run_legacy_fsdp_loss_norm_multinode.sh
#
# Environment variables you may override:
#   NNODES           -- number of nodes          (default: 1)
#   NPROC_PER_NODE   -- GPUs per node            (default: 2)
#   MINI_BATCH_SIZE  -- mini-batch size           (default: 4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

echo "=== Legacy FSDP smoke: ${NNODES} nodes x ${NPROC_PER_NODE} GPU ==="

exec torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv-backend=c10d \
    "${SCRIPT_DIR}/test_legacy_fsdp_loss_normalization.py"
