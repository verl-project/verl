#!/usr/bin/env bash

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${RUN_DIR}/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  export PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif command -v python >/dev/null 2>&1; then
  export PYTHON_BIN="$(command -v python)"
else
  export PYTHON_BIN="$(command -v python3)"
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

normalize_hf_hub_runtime() {
  if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" != "1" ]]; then
    return
  fi
  if "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec("hf_transfer") is not None else 1)
PY
  then
    return
  fi
  export HF_HUB_ENABLE_HF_TRANSFER=0
  log "hf_transfer is not installed; forcing HF_HUB_ENABLE_HF_TRANSFER=0 for this run"
}

normalize_hf_hub_runtime
