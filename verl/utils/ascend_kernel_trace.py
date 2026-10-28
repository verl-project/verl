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
"""Optional, bounded tracefs collection around a CI command; no mounts or kprobes.

The ring is host-wide, so unrelated events are possible. mm_page_alloc supplies
GFP flags and stacks but not the requested NUMA node. Trace PIDs may be host
PIDs invisible in the container. Save formats, clocks and overrun statistics;
do not infer absence of failures from an unavailable or overwritten trace.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

TRACE_ROOTS = (Path("/sys/kernel/tracing"), Path("/sys/kernel/debug/tracing"))


def status(output, **values):
    path = output / "status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**previous, **values, "updated_unix_ns": time.time_ns()}, indent=2))


def stop(output):
    """Only stop/remove the private instance recorded by this run; safe to repeat."""
    state = output / "instance.json"
    if not state.exists():
        return
    data = json.loads(state.read_text())
    root = Path(data["root"])
    name = data["name"]
    if root not in TRACE_ROOTS or not re.fullmatch(r"verl_mem_[0-9a-f]{32}", name):
        raise ValueError("Refusing to touch an unrecognized trace instance")
    instance = root / "instances" / name
    if not instance.exists():
        return
    # Disable even if output collection subsequently fails.
    (instance / "tracing_on").write_text("0")
    try:
        with (instance / "trace").open("rb") as source, (output / "trace.txt").open("wb") as target:
            shutil.copyfileobj(source, target)
        stats = {p.parent.name: p.read_text() for p in instance.glob("per_cpu/cpu*/stats")}
        (output / "buffer-stats.json").write_text(json.dumps(stats, indent=2))
        status(output, stopped=True, stop_monotonic_ns=time.monotonic_ns())
    finally:
        event = instance / "events/kmem/mm_page_alloc"
        if event.is_dir():
            (event / "enable").write_text("0")
            if "stacktrace" in (event / "trigger").read_text():
                (event / "trigger").write_text("!stacktrace")
        instance.rmdir()  # tracefs removes its virtual children; never recursive delete


def start(output):
    output.mkdir(parents=True, exist_ok=True)
    status(output, available=False, started=False)
    try:
        if os.sysconf("SC_PAGE_SIZE") != 4096:
            raise RuntimeError("Order-9 diagnostic requires 4 KiB base pages")
        root = next(p for p in TRACE_ROOTS if (p / "instances").is_dir())
        name = "verl_mem_" + uuid.uuid4().hex
        instance = root / "instances" / name
        # Record ownership before creating anything, for always-run cleanup.
        (output / "instance.json").write_text(json.dumps({"root": str(root), "name": name}))
        instance.mkdir()
        (instance / "tracing_on").write_text("0")
        (instance / "buffer_size_kb").write_text("128")  # per CPU, ~40 MiB on a 320-CPU host
        (instance / "trace_clock").write_text("mono")
        event = instance / "events/kmem/mm_page_alloc"
        (output / "event-format.txt").write_text((event / "format").read_text())
        condition = "order == 9 && pfn == 18446744073709551615"
        (event / "filter").write_text(condition)
        (event / "trigger").write_text("stacktrace if " + condition)
        (event / "enable").write_text("1")
        for filename, path in {
            "filter.txt": event / "filter",
            "trigger.txt": event / "trigger",
            "clock.txt": instance / "trace_clock",
            "buffer-total-kb.txt": instance / "buffer_total_size_kb",
        }.items():
            (output / filename).write_text(path.read_text())
        status(output, available=True, started=True, start_monotonic_ns=time.monotonic_ns())
        (instance / "tracing_on").write_text("1")
    except Exception as exc:
        status(output, available=False, error=repr(exc))
        try:
            stop(output)
        except Exception as cleanup:
            status(output, cleanup_error=repr(cleanup))
        print(f"Kernel tracing unavailable: {exc!r}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.stop:
        stop(output)
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a workload command is required after --")
    start(output)
    lock = threading.Lock()

    def finish():
        with lock:
            try:
                stop(output)
            except Exception as exc:
                print(f"Kernel trace cleanup failed: {exc!r}", flush=True)
                try:
                    status(output, cleanup_error=repr(exc))
                except OSError:
                    pass  # Preserve the workload result even if artifact storage is full.

    # Cap duration independently of workload progress. SIGKILL needs the CI
    # always-run cleanup step; loss of the entire runner still needs host cleanup.
    timer = threading.Timer(80 * 60, finish)
    timer.daemon = True
    timer.start()
    try:
        result = subprocess.call(command)
        return result if result >= 0 else 128 - result
    finally:
        timer.cancel()
        finish()


if __name__ == "__main__":
    raise SystemExit(main())
