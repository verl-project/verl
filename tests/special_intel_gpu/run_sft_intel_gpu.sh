#!/usr/bin/env bash
# SFT smoke test on Intel GPU — validates FSDP multi-GPU training without vLLM.
#
# Usage:
#   NUM_GPUS=4 bash tests/special_intel_gpu/run_sft_intel_gpu.sh

set -x

NUM_GPUS=${NUM_GPUS:-4}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/intel_gpu_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/intel_gpu_env.sh"
    configure_xpu_runtime sft
else
    echo "WARN: ${SCRIPT_DIR}/intel_gpu_env.sh not found; using built-in fallback env." >&2
    export CCL_ATL_SHM="${CCL_ATL_SHM:-1}"
    export CCL_BUFFER_CACHE="${CCL_BUFFER_CACHE:-0}"
    export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK="${CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK:-0}"
    export CCL_TOPO_ALGO="${CCL_TOPO_ALGO:-0}"
    export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-$(seq 0 $((NUM_GPUS - 1)) | paste -sd',')}"
fi

# Use python -m torch.distributed.run instead of torchrun so the correct
# Python interpreter is used regardless of PATH (e.g. inside conda envs).
PYTHON3=$(python3 -c "import torch; import sys; print(sys.executable)" 2>/dev/null || command -v python3)
$PYTHON3 -m torch.distributed.run --nproc-per-node=${NUM_GPUS} --standalone \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train_sft.parquet \
    data.val_files=$HOME/data/gsm8k/test_sft.parquet \
    data.train_batch_size=8 \
    data.max_length=1024 \
    model.path="${MODEL_PATH}" \
    model.trust_remote_code=True \
    model.use_remove_padding=False \
    +model.override_config.attn_implementation=sdpa \
    trainer.default_local_dir=./checkpoints/xpu_sft_test \
    trainer.project_name='verl_intel_gpu_sft_e2e' \
    trainer.experiment_name='qwen2_5_05b_intel_gpu_sft' \
    trainer.logger=console \
    trainer.total_epochs=1 \
    trainer.total_training_steps=5 \
    data.micro_batch_size_per_gpu=1 \
    trainer.save_freq=-1 \
    trainer.n_gpus_per_node=${NUM_GPUS} $@
