#!/usr/bin/env bash
# Megatron engine loss normalization — CP>1 smoke test.
#
# Validates that batch_num_tokens is all-reduced over DP×CP (not just DP)
# in the modern engine path (MegatronEngine.forward_backward_batch).
#
# Default: 1 node × 8 GPUs  (TP=1, PP=2, CP=2, DP=2)
#
# Usage:
#   bash tests/workers/engine/run_megatron_engine_cp_loss_norm.sh
#
# Override parallelism:
#   NPROC_PER_NODE=4 ACTOR_PP=1 ACTOR_CP=2 \
#       bash tests/workers/engine/run_megatron_engine_cp_loss_norm.sh
#
# Environment variables:
#   NPROC_PER_NODE   — GPUs per node            (default: 8)
#   ACTOR_TP         — tensor parallelism       (default: 1)
#   ACTOR_PP         — pipeline parallelism     (default: 2)
#   ACTOR_CP         — context parallelism      (default: 2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ACTOR_TP="${ACTOR_TP:-1}"
ACTOR_PP="${ACTOR_PP:-2}"
ACTOR_CP="${ACTOR_CP:-2}"

WORLD_SIZE=$((NPROC_PER_NODE))
DP_SIZE=$((WORLD_SIZE / (ACTOR_TP * ACTOR_PP * ACTOR_CP)))

echo "=== Megatron Engine CP loss normalization test ==="
echo "  ${NPROC_PER_NODE} GPUs: TP=${ACTOR_TP} PP=${ACTOR_PP} CP=${ACTOR_CP} DP=${DP_SIZE}"
echo "=================================================="

export ACTOR_PP ACTOR_TP ACTOR_CP

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$NPROC_PER_NODE" \
    "${SCRIPT_DIR}/test_special_megatron_engine_cp_loss_norm.py"
