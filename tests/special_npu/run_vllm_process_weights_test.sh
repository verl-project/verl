set -x

# Test Configuration
project_name='vllm-process-weights-test'
exp_name='test-vllm-process-weights-after-loading'

# Necessary env
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

# Model Weights Paths
MODEL_ID=${MODEL_ID:-moonshotai/Moonlight-16B-A3B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}

# Run the test
cd "$(dirname "$0")/../.." && python3 tests/workers/rollout/rollout_vllm/test_vllm_process_weights_after_loading.py
