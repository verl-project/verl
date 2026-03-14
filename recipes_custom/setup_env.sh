#!/usr/bin/env bash
# setup_env.sh — Set up environment for Qwen3.5 SFT (verl + megatron)
#
# Verified working versions:
#   mbridge:                git@34a87c4a (PR#83, qwen3.5 support)
#   megatron-core:          0.16.0
#   transformers:           5.2.0
#   torch:                  2.9.0+cu129
#   flash_attn:             2.8.1
#   flash-linear-attention: 0.4.1
#
# Usage:
#   bash setup_env.sh            # normal install
#   bash setup_env.sh --dry-run  # print commands only, do not execute
#
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] printing commands only, not executing"
fi

run() {
    if $DRY_RUN; then
        echo "[dry-run] $*"
    else
        echo ">>> $*"
        "$@"
    fi
}

echo "============================================"
echo " Qwen3.5 SFT environment setup"
echo "============================================"

# ----------------------------------------------------
# 1. mbridge — must install from git; PyPI 0.15.1 does
#    not register qwen3_5. Pinned to PR#83 merge commit.
# ----------------------------------------------------
echo ""
echo "[1/5] Installing mbridge (qwen3.5 support, commit 34a87c4a)..."
run pip install --no-deps \
    "git+https://github.com/ISEEKYAN/mbridge.git@34a87c4a83d5a54599d48e8552982575b066c2f8"

# ----------------------------------------------------
# 2. megatron-core 0.16.0
#    Required for attention_output_gate and GDN options.
# ----------------------------------------------------
echo ""
echo "[2/5] Installing megatron-core 0.16.0..."
run pip install --no-deps \
    "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.0"

# ----------------------------------------------------
# 3. transformers 5.2.0
# ----------------------------------------------------
echo ""
echo "[3/5] Installing transformers 5.2.0..."
run pip install "transformers==5.2.0"

# ----------------------------------------------------
# 4. flash-linear-attention 0.4.1
#    Required for Qwen3.5 GDN layers.
# ----------------------------------------------------
echo ""
echo "[4/5] Installing flash-linear-attention 0.4.1..."
run pip install "flash-linear-attention==0.4.1"

# ----------------------------------------------------
# 5. Install verl itself (no-deps; dependencies are
#    already provided by the base Docker image).
# ----------------------------------------------------
VERL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo ""
echo "[5/5] Installing verl (no-deps, editable): ${VERL_DIR}..."
run pip install --no-deps -e "${VERL_DIR}"

# ----------------------------------------------------
# Done — print installed versions
# ----------------------------------------------------
echo ""
echo "============================================"
echo " Done. Key package versions:"
echo "============================================"
if ! $DRY_RUN; then
    python -c "
import importlib

pkgs = {
    'torch':       'torch',
    'transformers':'transformers',
    'flash_attn':  'flash_attn',
    'fla':         'flash-linear-attention',
}
for mod, name in pkgs.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', 'unknown')
    except ImportError:
        ver = 'NOT INSTALLED'
    print(f'  {name:<28} {ver}')

try:
    import mbridge
    print(f'  {\"mbridge\":<28} {getattr(mbridge, \"__version__\", \"installed\")}')
except ImportError:
    print(f'  {\"mbridge\":<28} NOT INSTALLED')
"
fi
echo "============================================"
