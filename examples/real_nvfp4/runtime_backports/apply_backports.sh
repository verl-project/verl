#!/usr/bin/env bash
# Run only in a disposable build environment, never in a live training runtime.
set -euo pipefail
readonly BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly PYTHON_BIN=${PYTHON_BIN:-python}
"$PYTHON_BIN" "$BUNDLE/patch_megatron_fa4.py"
# Core 0.19.0 ships #5997; verify_runtime checks unchanged release bytes.
"$PYTHON_BIN" "$BUNDLE/apply_vllm_online_nvfp4_50029_50074.py"
"$PYTHON_BIN" "$BUNDLE/patch_te.py"
"$PYTHON_BIN" "$BUNDLE/patch_runtime.py"
"$PYTHON_BIN" "$BUNDLE/verify_runtime.py"
