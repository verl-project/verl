#!/usr/bin/env bash
# setup_env.sh — 配置 Qwen3.5 SFT (verl + megatron) 运行环境
#
# 验证可用的环境版本：
#   mbridge:                git@34a87c4a (PR#83, qwen3.5 支持)
#   megatron-core:          0.16.0
#   transformers:           5.2.0
#   torch:                  2.9.0+cu129
#   flash_attn:             2.8.1
#   flash-linear-attention: 0.4.1
#
# 用法：
#   bash setup_env.sh            # 正常安装
#   bash setup_env.sh --dry-run  # 只打印，不执行
#
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] 只打印命令，不执行"
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
echo " Qwen3.5 SFT 环境初始化"
echo "============================================"

# ----------------------------------------------------
# 1. mbridge — 必须从 git 装，PyPI 版本不含 qwen3_5
#    锁定 PR#83 merge commit: 34a87c4a
# ----------------------------------------------------
echo ""
echo "[1/5] 安装 mbridge (qwen3.5 支持，commit 34a87c4a)..."
run pip install --no-deps \
    "git+https://github.com/ISEEKYAN/mbridge.git@34a87c4a83d5a54599d48e8552982575b066c2f8"

# ----------------------------------------------------
# 2. megatron-core 0.16.0
#    attention_output_gate / GDN 选项需要此版本
# ----------------------------------------------------
echo ""
echo "[2/5] 安装 megatron-core 0.16.0..."
run pip install --no-deps \
    "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.0"

# ----------------------------------------------------
# 3. transformers 5.2.0
# ----------------------------------------------------
echo ""
echo "[3/5] 安装 transformers 5.2.0..."
run pip install "transformers==5.2.0"

# ----------------------------------------------------
# 4. flash-linear-attention 0.4.1
#    Qwen3.5 GDN 层依赖
# ----------------------------------------------------
echo ""
echo "[4/5] 安装 flash-linear-attention 0.4.1..."
run pip install "flash-linear-attention==0.4.1"

# ----------------------------------------------------
# 5. 安装 verl 本身（不装依赖，依赖镜像已有）
# ----------------------------------------------------
VERL_DIR="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "[5/5] 安装 verl (no-deps, editable): ${VERL_DIR}..."
run pip install --no-deps -e "${VERL_DIR}"

# ----------------------------------------------------
# 完成，打印版本信息
# ----------------------------------------------------
echo ""
echo "============================================"
echo " 安装完成，当前关键版本："
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
