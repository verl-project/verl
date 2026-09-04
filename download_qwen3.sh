#!/bin/bash
# ============================================================
# download_qwen3.sh
# 只负责下载模型到指定目录
# 用法: bash download_qwen3.sh [4B|8B|both]
# ============================================================
set -euo pipefail

TARGET="${1:-both}"
MODEL_SAVE_DIR="/ocean/projects/med230010p/yji3/models"

export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
export HF_HUB_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/hub
export TMPDIR=/ocean/projects/med230010p/yji3/.tmp

MODEL_4B_ID="Qwen/Qwen3-4B"
MODEL_8B_ID="Qwen/Qwen3-8B"

# ── log 必须写 stderr，否则被 $() 捕获污染返回值 ──────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

mkdir -p "$MODEL_SAVE_DIR"

check_deps() {
    python3 -c "import huggingface_hub" 2>/dev/null || {
        log "安装 huggingface_hub..."
        pip install huggingface_hub --quiet
    }
}

# 下载到 MODEL_SAVE_DIR，只输出最终路径到 stdout
download_model() {
    local model_id="$1"
    local target_dir="$MODEL_SAVE_DIR/$(echo "$model_id" | tr '/' '--')"

    log "=============================="
    log "开始下载: $model_id"
    log "目标目录: $target_dir"
    log "=============================="

    python3 - <<PYEOF
import sys
from huggingface_hub import snapshot_download

model_id = "$model_id"
target_dir = "$target_dir"

print(f"[下载] {model_id} -> {target_dir}", file=sys.stderr)
path = snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print(f"[完成] 本地路径: {path}", file=sys.stderr)
# 只输出路径到 stdout，供 shell 捕获
print(path)
PYEOF
}

check_deps

case "$TARGET" in
    4B)
        PATH_4B=$(download_model "$MODEL_4B_ID")
        log "Qwen3-4B 下载完成: $PATH_4B"
        echo "QWEN3_4B_PATH=$PATH_4B"
        ;;
    8B)
        PATH_8B=$(download_model "$MODEL_8B_ID")
        log "Qwen3-8B 下载完成: $PATH_8B"
        echo "QWEN3_8B_PATH=$PATH_8B"
        ;;
    both)
        log ">>> 下载 Qwen3-4B"
        PATH_4B=$(download_model "$MODEL_4B_ID")
        log "Qwen3-4B 下载完成: $PATH_4B"

        log ">>> 下载 Qwen3-8B"
        PATH_8B=$(download_model "$MODEL_8B_ID")
        log "Qwen3-8B 下载完成: $PATH_8B"

        # 输出路径供外部脚本 source 使用
        echo "QWEN3_4B_PATH=$PATH_4B"
        echo "QWEN3_8B_PATH=$PATH_8B"
        ;;
    *)
        echo "Unknown target: $TARGET (use 4B / 8B / both)" >&2
        exit 1
        ;;
esac

log "下载全部完成 ✓"