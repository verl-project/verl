#!/usr/bin/env bash
set -eo pipefail
W=/mnt/share/y00938198/kimi-k31/test_wxx_skill
if [[ -f "$W/scripts/env-resolved.sh" ]]; then source "$W/scripts/env-resolved.sh"; else source "$W/scripts/activate.sh"; fi
# Reboot clears the task-local tmpfs and short Ray socket path.
if [[ "$(findmnt -n -o FSTYPE -T "$W/cache/local")" != tmpfs || ! -d /tmp/k31-y00938198 ]]; then
    echo "Task-local cache mount/Ray path missing; restore them before launch" >&2
    exit 2
fi
/usr/bin/python3 "$W/scripts/check_idle.py"
export KIMI_BACKEND_ROOT="$W" MODEL_PATH="$W/model"
export VERL_EXPERIMENT_NAME=kimi31_4layer_alignment_423_$(date +%Y%m%d_%H%M%S)
export KIMI_RUN_TOKEN=kimi31-y00938198-423-$(cat /proc/sys/kernel/random/uuid)
export KIMI_ALIGNMENT_AUDIT_DIR="$W/results/$VERL_EXPERIMENT_NAME/audit"
export VERL_TOTAL_TRAINING_STEPS=2 TRAIN_BATCH_SIZE=8 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=8
export VERL_MAX_RESPONSE_LENGTH=64 VERL_VLLM_MAX_NUM_SEQS=16
export VERL_VLLM_ENFORCE_EAGER=True KIMI_ENABLE_R3=False
export KIMI_CPU_OFFLOAD=True KIMI_SWAP_ACTIVATION=True
export HCCL_OP_EXPANSION_MODE=AIV CPU_AFFINITY_CONF=0
export KIMI_ALIGNMENT_PREFIX_PROBE=1
export VERL_METRICS_GATHER_ON_NPU=1
mkdir -p "$KIMI_ALIGNMENT_AUDIT_DIR"
printf "%s\n" "$KIMI_RUN_TOKEN" > "$KIMI_ALIGNMENT_AUDIT_DIR/../run-token.txt"
if [[ "${KIMI_ALIGNMENT_RECURRENT_KDA:-0}" == 1 && "${KIMI_ALIGNMENT_V2_SCORING:-0}" == 1 ]]; then
    /usr/bin/python3 "$W/scripts/record_run.py" "$VERL_EXPERIMENT_NAME"
fi
exec bash "$W/scripts/run-alignment.sh" "$@"
