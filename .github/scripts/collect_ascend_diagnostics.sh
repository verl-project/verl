#!/usr/bin/env bash
# Capture host-side evidence without stopping processes or changing HCCL settings.
set -u

root="${ASCEND_PROCESS_LOG_PATH:-${GITHUB_WORKSPACE:-$PWD}/ci-ascend-diagnostics/ascend}"
root="${root%/ascend}"
mode="${1:-before}"
mkdir -p "$root"

capture() {
    local file="$1"
    shift
    timeout 15s "$@" > "$file" 2>&1 || true
}

snapshot() {
    local target="$root/$1"
    mkdir -p "$target"
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$target/time.txt"
    capture "$target/npu-smi.txt" npu-smi info
    capture "$target/processes.txt" ps -eo pid,ppid,stat,nlwp,etime,comm
    capture "$target/tcp-sockets.txt" ss -tanp
    capture "$target/udp-sockets.txt" ss -uanp
    cat /proc/meminfo > "$target/meminfo.txt"
    cat /proc/sysvipc/shm > "$target/sysvipc-shm.txt" 2>&1 || true
}

if [ "$mode" = monitor ]; then
    trap 'exit 0' TERM INT
    while true; do
        snapshot "samples/$(date -u +%Y%m%dT%H%M%S)"
        sleep 15 &
        wait "$!" || true
    done
fi

snapshot "$mode"
if [ "$mode" = before ]; then
    capture "$root/uname.txt" uname -a
    capture "$root/network-addresses.txt" ip -brief address
    capture "$root/network-routes.txt" ip route
    capture "$root/storage.txt" df -h /dev/shm /tmp "${GITHUB_WORKSPACE:-$PWD}"
    capture "$root/revision.txt" git rev-parse HEAD
    cat /usr/local/Ascend/driver/version.info > "$root/driver-version.txt" 2>&1 || true
    cat /usr/local/Ascend/ascend-toolkit/latest/*-linux/ascend_toolkit_install.info > "$root/cann-version.txt" 2>&1 || true
    python verl/utils/hccl_diagnostics.py ci.before > "$root/identity.txt" 2>&1 || true
    cat "$root/identity.txt"
fi

if [ "$mode" = after ]; then
    # Archive inside the job container; upload-artifact only needs the shared workspace.
    if [ -d /tmp/ray/session_latest/logs ]; then
        tar -czf "$root/ray-logs.tar.gz" -C /tmp/ray/session_latest logs || true
    fi
    if [ -d "$HOME/ascend/log" ]; then
        tar -czf "$root/ascend-default-logs.tar.gz" -C "$HOME/ascend" log || true
    fi
    if [ -f output.txt ]; then
        cp output.txt "$root/training-output.txt" || true
    fi
fi
