#!/usr/bin/env bash
# Run only in a disposable build environment, never in a live training runtime.
set -euo pipefail
readonly BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly PYTHON_BIN=${PYTHON_BIN:-python}
# TE imports Torch before the latter can choose its bundled NCCL. Prefer the
# selected environment's matching NCCL for every build-time patch/verification.
readonly NCCL_LIBDIR=$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl/lib")')
[[ -d "$NCCL_LIBDIR" ]]
export LD_LIBRARY_PATH="$NCCL_LIBDIR:${LD_LIBRARY_PATH:-}"
"$PYTHON_BIN" "$BUNDLE/patch_megatron_fa4.py"
# Core 0.19.0 ships #5997; verify_runtime checks unchanged release bytes.
"$PYTHON_BIN" "$BUNDLE/apply_vllm_online_nvfp4_50029_50074.py"
"$PYTHON_BIN" "$BUNDLE/patch_te.py"
"$PYTHON_BIN" "$BUNDLE/patch_runtime.py"
"$PYTHON_BIN" "$BUNDLE/verify_runtime.py"
