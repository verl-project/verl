#!/bin/bash
# Fix GPU/NCCL issues and run experiment
# For NVIDIA H100 GPUs

set -e

PHASE="${1:-phase1}"

echo "============================================================"
echo "  GPU/NCCL Fix & Run Script"
echo "  Phase: ${PHASE}"
echo "============================================================"

# ============================================================
# Step 1: Aggressively kill ALL Python/Ray processes
# ============================================================
echo ""
echo "[1/5] Killing all Python and Ray processes..."

# Kill Ray first
ray stop --force 2>/dev/null || true
sleep 2

# Kill all python processes (be careful if you have other important python jobs)
python_pids=$(pgrep -f "python" || true)
if [ -n "$python_pids" ]; then
    echo "  Found Python processes: $python_pids"
    echo "  Killing..."
    pgrep -f "python" | xargs -r kill -9 2>/dev/null || true
    sleep 3
fi

# Double check
remaining=$(pgrep -f "python" | wc -l)
echo "  Remaining Python processes: $remaining"

# ============================================================
# Step 2: Wait for GPU to fully release
# ============================================================
echo ""
echo "[2/5] Waiting for GPU to release..."

for i in 1 2 3 4 5; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "$used" -lt "500" ]; then
        echo "  GPU memory released (${used}MB used)"
        break
    fi
    echo "  GPU still in use (${used}MB), waiting... ($i/5)"
    sleep 5
done

# Reset GPU
echo "  Running nvidia-smi..."
nvidia-smi

# ============================================================
# Step 3: Set NCCL/CUDA environment variables for H100
# ============================================================
echo ""
echo "[3/5] Setting NCCL/CUDA environment for H100..."

# NCCL tuning for H100
export NCCL_IB_DISABLE=1           # Disable InfiniBand (single node)
export NCCL_P2P_LEVEL=NVL          # Use NVLink for P2P (H100 has NVLink)
export NCCL_DEBUG=WARN             # Only show warnings, not full info
export NCCL_ASYNC_ERROR_HANDLING=1 # Better error reporting

# CUDA tuning
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=0

# PyTorch distributed tuning
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Hugging Face offline mode (no network)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Ray tuning - limit concurrency to avoid H100 conflicts
export RAY_memory_monitor_refresh_ms=0

# Print env
echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "  NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"

# ============================================================
# Step 4: Verify GPU is clean
# ============================================================
echo ""
echo "[4/5] Verifying GPU state..."

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')
    
# Try simple tensor operation on each GPU
for i in range(torch.cuda.device_count()):
    try:
        x = torch.randn(100, 100, device=f'cuda:{i}')
        y = x @ x.T
        print(f'  GPU {i}: OK (test tensor passed)')
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  GPU {i}: FAILED - {e}')
        exit(1)

print('All GPUs ready!')
"

if [ $? -ne 0 ]; then
    echo "ERROR: GPU verification failed!"
    exit 1
fi

# ============================================================
# Step 5: Run experiment
# ============================================================
echo ""
echo "[5/5] Running experiment: ${PHASE}"
echo "============================================================"

# Run the experiment
cd "${WORKING_DIR:-/home/lingquh1xx/L2598/Temp/Spec-RL}"
bash run_experiment.sh "${PHASE}"
