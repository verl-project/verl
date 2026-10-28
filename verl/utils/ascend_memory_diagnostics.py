# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Opt-in Ascend allocation diagnostics; stdlib-only until installed in a vLLM worker.

Run this file directly to monitor the existing PPO command on a Linux CI node.
Snapshots add latency and can perturb allocation timing. Never alter allocation
arguments, retry a failed allocation, or suppress the original exception.
"""

import argparse
import datetime
import functools
import itertools
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

_offload_sequence = itertools.count()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"UNAVAILABLE: {exc}\n"


def thread_memory_policy():
    """Read this thread's default NUMA policy without changing it."""
    import ctypes

    try:
        library = ctypes.CDLL("libnuma.so.1", use_errno=True)
        query = library.get_mempolicy
        query.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        query.restype = ctypes.c_int
        possible = Path("/sys/devices/system/node/possible").read_text().strip()
        max_node = max(int(part.split("-")[-1]) for part in possible.split(",")) + 1
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        words = (max_node + bits - 1) // bits
        mask = (ctypes.c_ulong * words)()
        mode = ctypes.c_int()
        if query(ctypes.byref(mode), mask, words * bits, None, 0):
            raise OSError(ctypes.get_errno(), "get_mempolicy failed")
        return {"mode": mode.value, "nodes": [i for i in range(max_node) if mask[i // bits] & (1 << (i % bits))]}
    except Exception as exc:
        return {"unavailable": repr(exc)}


def snapshot(destination, detailed=True):
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "meminfo",
        "buddyinfo",
        "vmstat",
        "self/status",
        "thread-self/status",
        "thread-self/sched",
        "self/limits",
        "self/cgroup",
        "self/mountinfo",
    ):
        (destination / name.replace("/", "_")).write_text(read("/proc/" + name))
    if detailed:
        (destination / "pagetypeinfo").write_text(read("/proc/pagetypeinfo"))
        (destination / "self_numa_maps").write_text(read("/proc/self/numa_maps"))
        (destination / "thread-memory-policy.json").write_text(json.dumps(thread_memory_policy(), indent=2))
    for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
        (destination / (node.name + "_meminfo")).write_text(read(node / "meminfo"))
    memberships = [line.split(":", 2) for line in read("/proc/self/cgroup").splitlines() if line.count(":") == 2]
    groups = []
    for line in read("/proc/self/mountinfo").splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        if fs[0] not in ("cgroup", "cgroup2"):
            continue
        root, mount = fields[3:5]
        for _, controllers, group in memberships:
            if fs[0] == "cgroup2" and controllers:
                continue
            if fs[0] == "cgroup" and not (set(controllers.split(",")) & {"memory", "cpuset"} & set(fs[2].split(","))):
                continue
            relative = os.path.relpath(group, root)
            target = Path(mount) / relative if relative != ".." and not relative.startswith("../") else Path(mount)
            if not target.is_dir():
                target = Path(mount)
            while True:
                entry = {"path": str(target), "mount_root": root, "membership": group}
                for item in target.iterdir():
                    if item.name.startswith(("memory.", "cpuset.")) and item.is_file():
                        entry[item.name] = read(item)
                groups.append(entry)
                if target == Path(mount):
                    break
                target = target.parent
    (destination / "cgroups.json").write_text(json.dumps(groups, indent=2))
    (destination / "timestamp.txt").write_text(now() + "\nAncestors above cgroup mount root may be hidden.\n")


def capture(folder, name, command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        output = f"exit={result.returncode}\n{result.stdout}{result.stderr}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = f"UNAVAILABLE: {exc}"
    (folder / name).write_text(output)


def record(folder, detailed=None, **details):
    """Diagnostic failures must not replace a workload exception."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        details.update(
            time=now(),
            monotonic_ns=time.monotonic_ns(),
            pid=os.getpid(),
            ppid=os.getppid(),
            tid=threading.get_native_id(),
        )
        # Preserve the event/exception even if a later procfs read fails.
        (folder / "event.json").write_text(json.dumps(details, indent=2, default=str))
        snapshot(
            folder,
            detailed=folder.name in {"installed", "failed", "initial", "final"} if detailed is None else detailed,
        )
        (folder / "self_maps").write_text(read("/proc/self/maps"))
    except Exception as exc:
        print(f"Ascend memory diagnostic failed: {exc!r}", file=sys.stderr, flush=True)


def install_worker_init(worker_class):
    """Wrap the resolved worker method, without changing global tensor constructors."""
    original = worker_class.init_device
    if getattr(original, "_verl_memory_diagnostics", False):
        return

    @functools.wraps(original)
    def init_device(self, *args, **kwargs):
        root = os.environ.get("VERL_ASCEND_MEMORY_DIAGNOSTICS")
        if not root:
            return original(self, *args, **kwargs)
        folder = Path(root) / "worker_init" / str(os.getpid()) / str(time.monotonic_ns())
        record(folder / "before", detailed=True, method=original.__qualname__, stack=traceback.format_stack())
        started = time.monotonic()
        try:
            result = original(self, *args, **kwargs)
        except Exception:
            record(folder / "failed", exception=traceback.format_exc(), elapsed_s=time.monotonic() - started)
            try:
                capture(folder / "failed", "kernel.txt", ["dmesg", "-T"])
            except Exception as exc:
                print(f"Kernel capture failed: {exc!r}", file=sys.stderr, flush=True)
            raise
        record(folder / "after", elapsed_s=time.monotonic() - started)
        return result

    init_device._verl_memory_diagnostics = True
    worker_class.init_device = init_device


@contextmanager
def fsdp_offload(tensor, handle_index):
    """Observe an FSDP1 transfer boundary, not individual internal allocations.

    Metadata reads do not copy tensor data or synchronize the device. Elapsed
    time measures the host call only; the original transfer remains asynchronous.
    """
    root = os.environ.get("VERL_ASCEND_MEMORY_DIAGNOSTICS")
    if not root:
        yield
        return
    transfer = Path(root) / "fsdp_offloads" / str(os.getpid()) / f"transfer-{next(_offload_sequence):06d}"
    details = {"handle_index": handle_index, "target_device": "cpu", "non_blocking": True}
    try:
        details.update(
            source_device=str(tensor.device),
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            numel=tensor.numel(),
            tensor_bytes=tensor.numel() * tensor.element_size(),
            stack=traceback.format_stack(),
        )
    except Exception as exc:
        details["metadata_error"] = repr(exc)
    # tensor_bytes is not the allocator's rounded request size.
    record(transfer / "before", detailed=True, **details)
    started = time.monotonic()
    try:
        yield
    except Exception:
        record(transfer / "failed", exception=traceback.format_exc(), elapsed_s=time.monotonic() - started)
        try:
            capture(transfer / "failed", "kernel.txt", ["dmesg", "-T"])
        except Exception as exc:
            print(f"Kernel capture failed: {exc!r}", file=sys.stderr, flush=True)
        raise
    else:
        record(transfer / "after", elapsed_s=time.monotonic() - started)


def install():
    root = os.environ.get("VERL_ASCEND_MEMORY_DIAGNOSTICS")
    if not root:
        return
    import torch
    from vllm_ascend.device_allocator import camem

    if getattr(camem, "_verl_memory_diagnostics", False):
        return
    original = torch.empty
    sequence = itertools.count()
    worker = Path(root) / "workers" / str(os.getpid())

    @functools.wraps(original)
    def empty(*args, **kwargs):
        if not kwargs.get("pin_memory", False):
            return original(*args, **kwargs)
        allocation = worker / f"allocation-{next(sequence):06d}"
        request = {
            "args": repr(args),
            "kwargs": repr(kwargs),
            "stack": traceback.format_stack(),
        }
        record(allocation / "before", **request)
        started = time.monotonic()
        try:
            result = original(*args, **kwargs)
        except Exception:
            # Snapshot the failing process before expensive kernel-log collection.
            record(allocation / "failed", exception=traceback.format_exc(), elapsed_s=time.monotonic() - started)
            try:
                capture(allocation / "failed", "kernel.txt", ["dmesg", "-T"])
            except Exception as exc:
                print(f"Kernel capture failed: {exc!r}", file=sys.stderr, flush=True)
            raise
        record(allocation / "after", elapsed_s=time.monotonic() - started)
        return result

    class AllocatorTorchProxy:
        # Keep global torch.empty's identity intact: device contexts and Dynamo
        # rely on it. Only the allocator module's explicit calls are intercepted.
        def __getattr__(self, name):
            return empty if name == "empty" else getattr(torch, name)

    camem.torch = AllocatorTorchProxy()
    camem._verl_memory_diagnostics = True
    record(
        worker / "installed",
        torch_version=torch.__version__,
        hook="vllm_ascend.device_allocator.camem.torch.empty",
        global_torch_empty_unchanged=torch.empty is original,
        allocator_source=camem.__file__,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host-only", action="store_true", help="Sample host state without allocation interception")
    parser.add_argument(
        "--require-fsdp-offload", action="store_true", help="Require FSDP1 transfer evidence on success"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.host_only and args.require_fsdp_offload:
        parser.error("--require-fsdp-offload cannot be combined with --host-only")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a workload command is required after --")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["VERL_ASCEND_MEMORY_DIAGNOSTICS"] = "" if args.host_only else str(root)
    for name, cmd in {
        "uname.txt": ["uname", "-a"],
        "npu-smi.txt": ["npu-smi", "info"],
        "packages.txt": [sys.executable, "-m", "pip", "list"],
        "kernel-before.txt": ["dmesg", "-T"],
        "revision.txt": ["git", "rev-parse", "HEAD"],
        "source-hashes.txt": [
            "sha256sum",
            __file__,
            "tests/special_npu/run_qwen3_06b_ppo.sh",
            "verl/workers/rollout/vllm_rollout/utils.py",
            "verl/utils/fsdp_utils.py",
            "verl/utils/ascend_kernel_trace.py",
        ],
    }.items():
        capture(root, name, cmd)
    record(root / "initial", command=command)
    # Console output is captured by the workflow's tee; preserve child exit status.
    child = subprocess.Popen(command)
    sample = 0
    while child.poll() is None:
        record(root / "host" / f"sample-{sample:06d}")
        sample += 1
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    record(root / "final", exit_code=child.returncode)
    capture(root, "kernel-after.txt", ["dmesg", "-T"])
    installed = list(root.glob("workers/*/installed/event.json"))
    allocations = list(root.glob("workers/*/allocation-*/before/event.json"))
    failures = list(root.glob("workers/*/allocation-*/failed/event.json"))
    offloads = list(root.glob("fsdp_offloads/*/transfer-*/before/event.json"))
    offload_failures = list(root.glob("fsdp_offloads/*/transfer-*/failed/event.json"))
    offload_completed = list(root.glob("fsdp_offloads/*/transfer-*/after/event.json"))
    init_calls = list(root.glob("worker_init/*/*/before/event.json"))
    init_completed = list(root.glob("worker_init/*/*/after/event.json"))
    init_failures = list(root.glob("worker_init/*/*/failed/event.json"))
    sleep_allocations = []
    for event in allocations:
        stack = "".join(json.loads(event.read_text())["stack"])
        if "vllm_ascend" in stack and "sleep" in stack:
            sleep_allocations.append(str(event.relative_to(root)))
    summary = {
        "exit_code": child.returncode,
        "host_only": args.host_only,
        "instrumented_workers": len(installed),
        "pinned_allocation_requests": len(allocations),
        "failed_allocations": len(failures),
        "vllm_ascend_sleep_allocations": sleep_allocations,
        "fsdp_offload_processes": len({event.parents[2].name for event in offloads}),
        "fsdp_offload_calls": len(offloads),
        "fsdp_offload_completed": len(offload_completed),
        "fsdp_offload_failures": len(offload_failures),
        "fsdp_offload_failure_paths": [str(event.relative_to(root)) for event in offload_failures],
        "worker_init_calls": len(init_calls),
        "worker_init_completed": len(init_completed),
        "worker_init_failures": len(init_failures),
        "worker_init_failure_paths": [str(event.relative_to(root)) for event in init_failures],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({**summary, "vllm_ascend_sleep_allocations": len(sleep_allocations)}), flush=True)
    if child.returncode:
        return child.returncode if child.returncode > 0 else 128 - child.returncode
    if not args.host_only and (not init_calls or init_failures or len(init_completed) != len(init_calls)):
        print("Diagnostic incomplete: missing, failed or unfinished worker initialization", file=sys.stderr)
        return 2
    if args.require_fsdp_offload and (not offloads or offload_failures or len(offload_completed) != len(offloads)):
        print("Diagnostic incomplete: missing, failed or unfinished FSDP offload calls", file=sys.stderr)
        return 2
    if not args.host_only and not sleep_allocations:
        print("Diagnostic incomplete: no instrumented vllm_ascend sleep allocation observed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
