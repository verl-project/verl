# Copyright 2026 Individual Contributor: yue-zeng-yue
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
"""Run CUDA acceptance in bounded, separate torchrun processes; never CPU-fallback."""

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

DION_COMMIT = "7692479288f928a3d61d1cc58137b4dbe4f71000"
DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parent.parent


def dion_revision():
    """Require the pinned Dion commit and reject modified local dependency sources."""
    distribution = importlib.metadata.distribution("dion")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    revision = direct_url.get("vcs_info", {}).get("commit_id")
    if revision is None and direct_url.get("url", "").startswith("file:"):
        source = unquote(urlparse(direct_url["url"]).path)
        revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", source, "status", "--porcelain"], text=True).strip()
        if dirty:
            raise RuntimeError("Pinned Dion source has local modifications")
    if revision != DION_COMMIT:
        raise RuntimeError(f"Install the documented pinned Dion revision {DION_COMMIT}; found {revision!r}")
    return revision


def run_worker(case, precision, output, timeout, dynamo_recompile_limit=None):
    """Launch a bounded torchrun process and require its assertion-completion marker."""
    ranks = 1 if case == "smoke" else 2
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={ranks}",
        "--max_restarts=0",
        str(DIRECTORY / "muon_fsdp2_acceptance_worker.py"),
        "--case",
        case,
        "--precision",
        precision,
        "--output",
        str(output),
    ]
    if dynamo_recompile_limit is not None:
        command.extend(["--dynamo-recompile-limit", str(dynamo_recompile_limit)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    log_path = output / f"{case}.log"
    print(f"Launching {case}/{precision}, {ranks} CUDA rank(s), timeout={timeout}s; log={log_path}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=REPOSITORY, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            status = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f"{case}/{precision} timed out or was interrupted; see {log_path}") from None
    content = log_path.read_text()
    print(content, end="", flush=True)
    if status:
        raise RuntimeError(f"{case}/{precision} exited {status}; see {log_path}")
    marker = f"MUON_FSDP2_WORKER_OK case={case} precision={precision}"
    if marker not in content.splitlines():
        raise RuntimeError(f"{case}/{precision} exited without its assertion-completion marker")
    return {"case": case, "precision": precision, "exit_code": status, "log": str(log_path)}


def main():
    """Validate the GPU environment, run ordered phases and save their result manifest."""
    if not __debug__:
        raise RuntimeError("Acceptance must run with assertions enabled; do not use python -O/PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["smoke", "alignment", "optimizer", "training", "checkpoint", "all"], default="all"
    )
    parser.add_argument("--precision", choices=["fp32", "bf16", "both"], default="both")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout in seconds for each torchrun process")
    parser.add_argument(
        "--dynamo-recompile-limit",
        type=int,
        help="Explicit compiler cache budget for the paired sharded/reference paths; default keeps PyTorch settings",
    )
    parser.add_argument(
        "--output", type=Path, help="New or empty artifact directory; defaults to a fresh temp directory"
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.dynamo_recompile_limit is not None and args.dynamo_recompile_limit <= 0:
        parser.error("--dynamo-recompile-limit must be positive")
    import torch

    required = 1 if args.case == "smoke" else 2
    if not torch.cuda.is_available() or torch.cuda.device_count() < required:
        raise RuntimeError(
            f"NOT RUN: {args.case} requires at least {required} visible CUDA GPUs; "
            f"found {torch.cuda.device_count()}. Standalone acceptance never skips or substitutes CPU."
        )
    revision = dion_revision()
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="verl-muon-acceptance-"))
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Artifact directory must be empty to exclude stale results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    precisions = ["fp32", "bf16"] if args.precision == "both" else [args.precision]
    cases = ["smoke", "alignment", "continuous", "save", "resume"] if args.case == "all" else [args.case]
    if args.case == "checkpoint":
        cases = ["continuous", "save", "resume"]
    results = []
    for precision in precisions:
        precision_output = output / precision
        precision_output.mkdir()
        for case in cases:
            # Synchronous wait above guarantees save5's process has EXITED before
            # resume constructs fresh model, optimizer, scheduler and NCCL group.
            results.append(run_worker(case, precision, precision_output, args.timeout, args.dynamo_recompile_limit))
    manifest = {
        "status": "passed",
        "case": args.case,
        "precision": args.precision,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(index) for index in range(required)],
        "dion_commit": revision,
        "dynamo_recompile_limit_override": args.dynamo_recompile_limit,
        "results": results,
    }
    (output / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"MUON_FSDP2_ACCEPTANCE_OK case={args.case} precision={args.precision} artifacts={output}", flush=True)


if __name__ == "__main__":
    main()
