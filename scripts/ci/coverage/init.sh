#!/usr/bin/env bash
set -euo pipefail

command -v uv >/dev/null || python -m pip install uv
uv pip install --python "$(python -c 'import sys; print(sys.executable)')" "coverage==7.8.2" "PyYAML==6.0.2"
exec python scripts/ci/coverage/collect.py init
