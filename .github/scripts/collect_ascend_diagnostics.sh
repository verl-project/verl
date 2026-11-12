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
    local status=0
    timeout 15s "$@" > "$file" 2>&1 || status=$?
    printf '[VERL_ASCEND_CAPTURE] exit_code=%s command=%s\n' "$status" "$*" >> "$file"
}

emit_file() {
    local file="$1"
    echo "[VERL_ASCEND_SNAPSHOT] file=$file lines=$(wc -l < "$file")"
    sed 's/^/[VERL_ASCEND_SNAPSHOT] /' "$file"
}

emit_snapshot() {
    local target="$1"
    echo "[VERL_ASCEND_SNAPSHOT] BEGIN $target"
    for name in time npu-smi processes tcp-sockets udp-sockets meminfo sysvipc-shm; do
        emit_file "$target/$name.txt"
    done
    echo "[VERL_ASCEND_SNAPSHOT] END $target"
}

snapshot() {
    local target="$root/$1"
    mkdir -p "$target"
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$target/time.txt"
    capture "$target/npu-smi.txt" npu-smi info
    capture "$target/processes.txt" ps -eo pid,ppid,stat,nlwp,etime,comm
    if command -v ss >/dev/null 2>&1; then
        capture "$target/tcp-sockets.txt" ss -tanp
        capture "$target/udp-sockets.txt" ss -uanp
    else
        capture "$target/tcp-sockets.txt" python verl/utils/hccl_diagnostics.py ci.network tcp
        capture "$target/udp-sockets.txt" python verl/utils/hccl_diagnostics.py ci.network udp
    fi
    cat /proc/meminfo > "$target/meminfo.txt"
    cat /proc/sysvipc/shm > "$target/sysvipc-shm.txt" 2>&1 || true
}

if [ "$mode" = monitor ]; then
    trap 'exit 0' TERM INT
    while true; do
        sample="samples/$(date -u +%Y%m%dT%H%M%S)"
        snapshot "$sample"
        # Preserve live device/process evidence even if a timeout prevents the final step.
        emit_snapshot "$root/$sample"
        sleep 15 &
        wait "$!" || true
    done
fi

snapshot "$mode"
if [ "$mode" = before ]; then
    capture "$root/uname.txt" uname -a
    if command -v ip >/dev/null 2>&1; then
        capture "$root/network-addresses.txt" ip -brief address
        capture "$root/network-routes.txt" ip route
    else
        capture "$root/network-addresses.txt" python verl/utils/hccl_diagnostics.py ci.network interfaces
        capture "$root/network-routes.txt" python verl/utils/hccl_diagnostics.py ci.network routes
    fi
    capture "$root/storage.txt" df -h /dev/shm /tmp "${GITHUB_WORKSPACE:-$PWD}"
    capture "$root/revision.txt" git rev-parse HEAD
    cat /usr/local/Ascend/driver/version.info > "$root/driver-version.txt" 2>&1 || true
    cat /usr/local/Ascend/ascend-toolkit/latest/*-linux/ascend_toolkit_install.info > "$root/cann-version.txt" 2>&1 || true
    python verl/utils/hccl_diagnostics.py ci.baseline || true
    python verl/utils/hccl_diagnostics.py ci.before > "$root/identity.txt" 2>&1 || true
    cat "$root/identity.txt"
    for name in uname network-addresses network-routes storage revision driver-version cann-version; do
        emit_file "$root/$name.txt"
    done
    emit_snapshot "$root/before"
fi

if [ "$mode" = after ]; then
    # Console output is sufficient for users who cannot access the runner or artifacts.
    emit_snapshot "$root/after"
    capture "$root/kernel-errors.txt" dmesg --ctime --level=err,warn
    emit_file "$root/kernel-errors.txt"
    python verl/utils/hccl_diagnostics.py ci.report || true
fi
