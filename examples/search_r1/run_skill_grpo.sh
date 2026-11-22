#!/usr/bin/env bash
# The rollout/retriever and GRPO settings are inherited from the baseline.
set -euo pipefail
RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export RUN_DIR=${RUN_DIR:-$RECIPE_DIR/../../../outputs/search-r1-skills-$(date +%Y%m%d-%H%M%S)}
exec bash "$RECIPE_DIR/run_grpo.sh" \
    +skill_distillation=search_r1 \
    trainer.project_name=search-r1-skills \
    trainer.experiment_name=search-r1-three-skill-teachers \
    "$@"
