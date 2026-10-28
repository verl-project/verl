"""Manual Linux/NPU diagnostic; run only on an Ascend CI node, never locally.

Fresh processes isolate allocator caches between scenarios. No model is loaded.
Artifacts include host snapshots and all visible memory-cgroup ancestors.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"UNAVAILABLE: {exc}\n"


def snapshot(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "meminfo",
        "buddyinfo",
        "pagetypeinfo",
        "vmstat",
        "self/status",
        "self/limits",
        "self/cgroup",
        "self/mountinfo",
    ):
        (destination / name.replace("/", "_")).write_text(read("/proc/" + name))
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


def worker(args):
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    phase = "initialization"
    result = {
        "pid": os.getpid(),
        "device": args.device,
        "pinned": args.pinned,
        "sizes_mib": args.sizes,
        "started": now(),
    }
    try:
        snapshot(folder / "before")
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        torch.npu.init()
        result["torch"] = torch.__version__
        result["torch_npu"] = torch_npu.__version__
        (folder / "ready").write_text(now())
        phase = "barrier"
        deadline = time.monotonic() + args.timeout
        while not Path(args.gate).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Allocation barrier timed out")
            time.sleep(0.05)
        held = []
        for index, size in enumerate(args.sizes):
            phase = f"allocation[{index}] {size} MiB"
            print(json.dumps({"time": now(), "phase": phase, "pinned": args.pinned}), flush=True)
            tensor = torch.empty(size * 1024 * 1024, dtype=torch.uint8, device="cpu", pin_memory=args.pinned)
            phase = f"touch[{index}] {size} MiB"
            tensor.fill_(1)  # Commit ordinary CPU pages too, for a meaningful control.
            held.append(tensor)
            print(json.dumps({"time": now(), "completed_mib": size, "is_pinned": tensor.is_pinned()}), flush=True)
        phase = "holding"
        snapshot(folder / "allocated")
        time.sleep(2)
        result["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - preserve any worker failure as a diagnostic artifact
        result.update(status="failed", phase=phase, error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result["finished"] = now()
        (folder / "result.json").write_text(json.dumps(result, indent=2))
        snapshot(folder / "after")
    return 0 if result.get("status") == "passed" else 1


def capture(folder, name, command):
    try:
        run = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        output = f"exit={run.returncode}\n{run.stdout}\n{run.stderr}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = f"UNAVAILABLE: {exc}"
    (folder / name).write_text(output)


def main(args):
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    snapshot(folder / "initial")
    for name, command in {
        "uname.txt": ["uname", "-a"],
        "npu-smi.txt": ["npu-smi", "info"],
        "packages.txt": [sys.executable, "-m", "pip", "list"],
        "kernel-before.txt": ["dmesg", "-T"],
        "revision.txt": ["git", "rev-parse", "HEAD"],
    }.items():
        capture(folder, name, command)
    summary = []
    # Max live tensors per worker in the mixed case: 1.25 GiB; eight workers: 10 GiB.
    scenarios = [
        ("cpu-512", False, 1, [512]),
        ("pinned-32", True, 1, [32]),
        ("pinned-512", True, 1, [512]),
        ("parallel-512", True, args.workers, [512]),
        ("parallel-mixed", True, args.workers, [512] + [32] * 8 + [512]),
    ]
    for repeat in range(args.repeats):
        for name, pinned, count, sizes in scenarios:
            stage = folder / f"{repeat:02d}-{name}"
            snapshot(stage / "before")
            gate = stage / "go"
            processes, handles = [], []
            started = time.monotonic()
            try:
                for device in range(count):
                    out = stage / f"worker-{device}"
                    out.mkdir()
                    handle = (out / "stdout.log").open("w")
                    handles.append(handle)
                    command = [
                        sys.executable,
                        "-u",
                        __file__,
                        "--worker",
                        "--device",
                        str(device),
                        "--output",
                        str(out),
                        "--gate",
                        str(gate),
                        "--timeout",
                        str(args.timeout),
                        "--sizes",
                        *map(str, sizes),
                    ]
                    if pinned:
                        command.append("--pinned")
                    processes.append(
                        subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    )
                timed_out = False
                sample = 0
                while any(p.poll() is None for p in processes):
                    if not gate.exists() and all(
                        (stage / f"worker-{i}" / "ready").exists() or p.poll() is not None
                        for i, p in enumerate(processes)
                    ):
                        gate.write_text(now())
                    if time.monotonic() - started > args.timeout:
                        timed_out = True
                        break
                    snapshot(stage / f"sample-{sample:04d}")
                    sample += 1
                    time.sleep(1)
            finally:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                for handle in handles:
                    handle.close()
            snapshot(stage / "after")
            summary.append(
                {
                    "scenario": stage.name,
                    "timed_out": timed_out,
                    "exit_codes": [p.returncode for p in processes],
                    "results": [
                        json.loads(read(stage / f"worker-{i}" / "result.json"))
                        if (stage / f"worker-{i}" / "result.json").exists()
                        else {
                            "status": "no_result",
                            "note": "Inspect stdout and exit code; may be killed or timed out.",
                        }
                        for i in range(count)
                    ],
                }
            )
            (folder / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary[-1]), flush=True)
    capture(folder, "kernel-after.txt", ["dmesg", "-T"])
    return int(any(s["timed_out"] or any(s["exit_codes"]) for s in summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gate")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 512])
    parser.add_argument("--pinned", action="store_true")
    arguments = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This diagnostic must run on Linux with Ascend NPUs")
    sys.exit(worker(arguments) if arguments.worker else main(arguments))
