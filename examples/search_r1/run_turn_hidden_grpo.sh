#!/usr/bin/env bash
# Add turn-state diversity to the existing skill recipe; CLI overrides stay last.
set -euo pipefail
RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The encoder shares the first retrieval-visible GPU; keep FP32 in the backend.
export RETRIEVAL_ENCODER_DEVICE=${RETRIEVAL_ENCODER_DEVICE:-cuda:0}
TEACHER_MICRO_BATCH_SIZE=${TEACHER_MICRO_BATCH_SIZE:-32}
exec bash "$RECIPE_DIR/run_skill_grpo.sh" \
    +turn_hidden_diversity=search_r1 \
    skill_distillation.teacher_micro_batch_size_per_gpu="$TEACHER_MICRO_BATCH_SIZE" \
    turn_hidden_diversity.enabled=true \
    turn_hidden_diversity.margin=0.1 \
    turn_hidden_diversity.coefficient=0.001 \
    "$@"
