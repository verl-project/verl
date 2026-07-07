#!/bin/bash
# example: bash OPD/model/download_hf_model.sh Qwen/Qwen3-0.6B OPD/model/Qwen3-0.6B
# 检查参数数量是否正确
if [ "$#" -ne 2 ]; then
    echo "❌ 错误: 参数数量不匹配。"
    echo "💡 用法: $0 <模型名称> <下载路径>"
    echo "📌 示例: $0 Qwen/Qwen3-0.6B ./downloaded_models"
    exit 1
fi

MODEL_NAME=$1
SAVE_PATH=$2

echo "========================================="
echo "📦 模型名称: $MODEL_NAME"
echo "📂 下载/缓存路径: $SAVE_PATH"
echo "========================================="

# 可以在此处取消注释以使用国内 HuggingFace 镜像源加速下载
# export HF_ENDPOINT="https://hf-mirror.com"

# 使用 EOF 将 Python 代码嵌入在 Shell 脚本中运行
python3 - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "${MODEL_NAME}"
cache_path = "${SAVE_PATH}"

print(f"🔄 正在下载/验证 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)

print(f"🔄 正在下载/验证 Model 权重 (这可能需要一些时间)...")
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path)

print("✅ 下载完成！")
EOF