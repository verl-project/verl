#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn}"
ALLOW_FLASH_ATTN_SOURCE_BUILD="${ALLOW_FLASH_ATTN_SOURCE_BUILD:-0}"
MAX_JOBS="${MAX_JOBS:-4}"
FLASH_ATTN_BUILD_ROOT="${FLASH_ATTN_BUILD_ROOT:-${PROJECT_ROOT}/.cache/flash-attn-build}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${FLASH_ATTN_BUILD_ROOT}/pip-cache}"
export TMPDIR="${TMPDIR:-${FLASH_ATTN_BUILD_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"

mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

log "install_flash_attn: PYTHON_BIN=${PYTHON_BIN}"
log "install_flash_attn: FLASH_ATTN_SPEC=${FLASH_ATTN_SPEC}"
log "install_flash_attn: PIP_CACHE_DIR=${PIP_CACHE_DIR}"
log "install_flash_attn: TMPDIR=${TMPDIR}"

"${PYTHON_BIN}" - <<'PY'
import re
import sys

if not sys.platform.startswith("linux"):
    raise SystemExit("flash-attn install script currently supports Linux targets only.")

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch must be installed before running ./run/install_flash_attn.sh.") from exc

match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", torch.__version__)
if match is None:
    raise SystemExit(f"Could not parse installed torch version: {torch.__version__}")
version = tuple(int(part) for part in match.groups())
if version < (2, 8, 0):
    raise SystemExit(
        "reliable-gsm8k-builder requires PyTorch >= 2.8.0 for local Transformers backends. "
        f"Found torch=={torch.__version__}."
    )
print(f"torch ok: version={torch.__version__}")
PY

log "Trying wheel-only install first to avoid a RAM-heavy source build"
if "${PYTHON_BIN}" -m pip install --only-binary=:all: --prefer-binary "${FLASH_ATTN_SPEC}"; then
  "${PYTHON_BIN}" -c "import flash_attn; print(f'flash_attn ok: version={flash_attn.__version__}')"
  exit 0
fi

if [[ "${ALLOW_FLASH_ATTN_SOURCE_BUILD}" != "1" ]]; then
  cat >&2 <<EOF
No compatible prebuilt flash-attn wheel was found for this environment.

The script stopped here on purpose so pip would not fall back to a RAM-heavy source build.

If you want to allow a constrained source build, rerun:

  ALLOW_FLASH_ATTN_SOURCE_BUILD=1 MAX_JOBS=${MAX_JOBS} ./run/install_flash_attn.sh

The official flash-attn docs note that limiting MAX_JOBS helps avoid exhausting RAM on machines
with many CPU cores.
EOF
  exit 1
fi

log "Falling back to a constrained source build with MAX_JOBS=${MAX_JOBS}"
"${PYTHON_BIN}" -m pip install packaging psutil ninja
MAX_JOBS="${MAX_JOBS}" "${PYTHON_BIN}" -m pip install --no-cache-dir --no-build-isolation "${FLASH_ATTN_SPEC}"
"${PYTHON_BIN}" -c "import flash_attn; print(f'flash_attn ok: version={flash_attn.__version__}')"
