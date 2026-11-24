#!/usr/bin/env bash
# EM GRPO with adjacent-turn hidden-state diversity, without skill teachers.
set -euo pipefail
RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export RETRIEVAL_ENCODER_DEVICE=${RETRIEVAL_ENCODER_DEVICE:-cuda:0}
exec bash "$RECIPE_DIR/run_grpo.sh" \
    +turn_hidden_diversity=search_r1 \
    turn_hidden_diversity.enabled=true \
    turn_hidden_diversity.margin=0.1 \
    turn_hidden_diversity.coefficient=0.001 \
    "$@"
