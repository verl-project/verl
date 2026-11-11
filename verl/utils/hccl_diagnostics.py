# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in, per-rank HCCL initialization diagnostics without additional collectives."""

import hashlib
import json
import os
import resource
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_ENVIRONMENT_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "DIST_INIT_METHOD",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "ASCEND_DEVICE_ID",
    "DEVICE_ID",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "HCCL_HOST_SOCKET_PORT_RANGE",
    "HCCL_NPU_SOCKET_PORT_RANGE",
    "HCCL_IF_IP",
    "HCCL_SOCKET_IFNAME",
    "HCCL_INTRA_PCIE_ENABLE",
    "HCCL_INTRA_ROCE_ENABLE",
    "HCCL_CONNECT_TIMEOUT",
    "HCCL_EXEC_TIMEOUT",
    "HCCL_BUFFSIZE",
    "HCCL_DETERMINISTIC",
    "ASCEND_GLOBAL_LOG_LEVEL",
    "ASCEND_PROCESS_LOG_PATH",
    "OMP_NUM_THREADS",
    "VERL_TRAINING_WORKER_OMP_NUM_THREADS",
    "RUNNER_NAME",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_JOB",
    "NODE_NAME",
    "POD_NAME",
    "POD_NAMESPACE",
)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return f"unavailable: {error}"


def print_hccl_diagnostics(stage, **details):
    """Print a flushed JSON record and save it separately from Ray's log stream."""
    if os.environ.get("VERL_HCCL_DIAGNOSTICS") != "1":
        return
    try:
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "env": {key: os.environ[key] for key in _ENVIRONMENT_KEYS if key in os.environ},
            "cgroup": _read("/proc/self/cgroup"),
            "process": [
                line
                for line in _read("/proc/self/status").splitlines()
                if line.split(":", 1)[0] in {"NSpid", "Threads", "VmRSS", "VmLck", "Cpus_allowed_list"}
            ],
            "limits": {
                name: resource.getrlimit(getattr(resource, name))
                for name in ("RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_MEMLOCK")
            },
            "open_fds": len(os.listdir("/proc/self/fd")),
            **details,
        }
        boot_id = _read("/proc/sys/kernel/random/boot_id")
        record["host_boot_id_hash"] = (
            hashlib.sha256(boot_id.encode()).hexdigest()[:16] if not boot_id.startswith("unavailable:") else boot_id
        )
        record["namespaces"] = {name: os.readlink(f"/proc/self/ns/{name}") for name in ("pid", "net", "ipc", "mnt")}
        record["device_nodes"] = {
            str(path): {"major": os.major(path.stat().st_rdev), "minor": os.minor(path.stat().st_rdev)}
            for path in sorted(Path("/dev").glob("davinci[0-9]*"))
        }
        ray = sys.modules.get("ray")
        if ray is not None and ray.is_initialized():
            context = ray.get_runtime_context()
            record["ray"] = {"node_id": context.get_node_id(), "accelerator_ids": context.get_accelerator_ids()}
        torch = sys.modules.get("torch")
        if torch is not None:
            record["distributed_initialized"] = torch.distributed.is_initialized()
            # Do not initialize an NPU just to print diagnostics before communicator setup.
            npu = getattr(torch, "npu", None)
            if npu is not None and npu.is_initialized():
                record["npu"] = {
                    "current_device": npu.current_device(),
                    "memory_allocated": npu.memory_allocated(),
                    "memory_reserved": npu.memory_reserved(),
                }
        line = json.dumps(record, sort_keys=True, default=str)
        print(f"[VERL_HCCL_DIAGNOSTICS] {line}", flush=True)
        if directory := os.environ.get("ASCEND_PROCESS_LOG_PATH"):
            path = Path(directory) / "worker-stages"
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"{socket.gethostname()}-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(line + "\n")
    except Exception as error:
        # Diagnostic collection must not replace a training exception or stop initialization.
        try:
            print(f"[VERL_HCCL_DIAGNOSTICS] collection failed at {stage}: {error!r}", flush=True)
        except Exception:
            pass


@contextmanager
def hccl_diagnostic_stage(stage, **details):
    """Record stage entry, return, or the original exception without synchronizing ranks."""
    if os.environ.get("VERL_HCCL_DIAGNOSTICS") != "1":
        yield
        return
    started = time.monotonic()
    print_hccl_diagnostics(stage, status="begin", **details)
    try:
        yield
    except Exception as error:
        print_hccl_diagnostics(
            stage,
            status="error",
            elapsed_s=time.monotonic() - started,
            exception_type=type(error).__name__,
            exception=str(error),
            **details,
        )
        raise
    else:
        print_hccl_diagnostics(stage, status="success", elapsed_s=time.monotonic() - started, **details)


if __name__ == "__main__":
    print_hccl_diagnostics(sys.argv[1] if len(sys.argv) > 1 else "ci.snapshot")
