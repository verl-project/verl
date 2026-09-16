#!/usr/bin/env bash
# Run only in a disposable build environment, never in a live training runtime.
set -euo pipefail
readonly BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly PYTHON_BIN=${PYTHON_BIN:-python}
"$PYTHON_BIN" "$BUNDLE/patch_megatron_fa4.py"
"$PYTHON_BIN" "$BUNDLE/patch_megatron_checkpoint.py"
"$PYTHON_BIN" "$BUNDLE/apply_vllm_online_nvfp4_50029_50074.py"
"$PYTHON_BIN" "$BUNDLE/patch_te.py"
"$PYTHON_BIN" "$BUNDLE/patch_runtime.py"
"$PYTHON_BIN" "$BUNDLE/verify_runtime.py"
