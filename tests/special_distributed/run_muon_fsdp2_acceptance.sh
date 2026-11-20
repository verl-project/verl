#!/usr/bin/env bash
# CUDA-only entry point; the Python driver bounds every torchrun process and
# kills its entire process group on timeout. All errors propagate to this shell.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python "${repo_root}/tests/special_distributed/run_muon_fsdp2_acceptance.py" "$@"
