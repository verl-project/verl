#!/usr/bin/env bash
# Finish-hook command for the profiler e2e tests: drops a marker file so the calling script can
# assert that workers really executed the hook.
#
# The marker goes into a dedicated finish_hook_markers/ sub-directory of the profiler save_path,
# not save_path itself. Its name embeds the raw worker role (e.g. "actor_rollout_ref"), and that
# "_rollout_" substring would otherwise make test_check_profiler_output.py mistake the marker for a
# "*_rollout_*" profiler stage deliverable and fail the run -- the real trace files dodge this only
# because their role is sanitized to "actor-rollout-ref". Nesting the marker one level down hides it
# from the checker's top-level stage globs regardless of the role name.
set -euo pipefail

: "${VERL_PROFILE_SAVE_PATH:?must be exported by the profiler finish hook}"

MARKER_DIR="$VERL_PROFILE_SAVE_PATH/finish_hook_markers"
mkdir -p "$MARKER_DIR"
MARKER="$MARKER_DIR/finish_hook_ran_${VERL_PROFILE_ROLE:-unknown}_rank${VERL_PROFILE_RANK:-unknown}"
touch "$MARKER"
echo "profiler finish hook marker: $MARKER"
