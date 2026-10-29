# Copyright 2025 Individual Contributor: Steven Zhang
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
"""Read-only Ascend environment snapshot. Run before training, using its Python/env."""

import argparse
import fnmatch
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

MODULES = (
    "torch",
    "torch_npu",
    "acl",
    "transformers",
    "vllm",
    "vllm_ascend",
    "ray",
    "modelopt",
    "mindspeed",
    "safetensors",
    "liger_kernel",
    "triton",
    "tensordict",
    "numpy",
    "verl",
)
SECRET = re.compile(r"TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY|API_KEY", re.I)
PREFIXES = (
    "ASCEND",
    "HCCL",
    "NCCL",
    "CUDA",
    "CANN",
    "TORCH",
    "PYTORCH",
    "VLLM",
    "VERL",
    "RAY",
    "OMP",
    "MKL",
    "OPENBLAS",
    "NUMEXPR",
    "MALLOC",
    "NPU",
    "ACL",
    "HF_",
    "TRANSFORMERS",
    "ATB",
    "GE_",
    "TBE_",
    "HCOM_",
    "MS_",
)
EXACT = {
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LIBRARY_PATH",
    "CPATH",
    "TASK_QUEUE_ENABLE",
    "COMBINED_ENABLE",
    "DYNAMIC_OP",
    "CPU_AFFINITY_CONF",
    "TOKENIZERS_PARALLELISM",
    "PYTHONHASHSEED",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "LANG",
    "LC_ALL",
}
EXACT.update({"MODEL_PATH", "TRAIN_FILE", "TEST_FILE", "NDEVICES_PER_NODE", "NNODES", "SP_SIZE"})
EXACT.update(
    {
        "INF_NAN_MODE_ENABLE",
        "ALLOW_CONV_HF32",
        "ALLOW_MATMUL_HF32",
        "PTCOPY_ENABLE",
        "MULTI_STREAM_MEMORY_REUSE",
        "DYNAMIC_COMPILE_ENABLE",
        "OP_DEBUG_LEVEL",
    }
)


def scrub(text):
    text = str(text)
    for key, value in os.environ.items():
        if SECRET.search(key) and len(value) >= 8:
            text = text.replace(value, "<redacted>")

    def clean_url(match):
        try:
            u = urlsplit(match.group())
            host = u.netloc.rsplit("@", 1)[-1]
            return urlunsplit((u.scheme, host, u.path, "", ""))
        except ValueError:
            return "<redacted-url>"

    return re.sub(r"(?:https?|ssh)://[^\s\"'<>]+", clean_url, text)


def dump(path, value):
    path.write_text(scrub(json.dumps(value, ensure_ascii=False, indent=2, default=str)) + "\n")


def command(argv, timeout=30, cwd=None):
    try:
        p = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, stderr = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            stdout, stderr = p.communicate()
            return {
                "argv": argv,
                "error": f"timeout after {timeout}s",
                "stdout": scrub(stdout),
                "stderr": scrub(stderr),
            }
        return {"argv": argv, "returncode": p.returncode, "stdout": scrub(stdout), "stderr": scrub(stderr)}
    except OSError as exc:
        return {"argv": argv, "error": scrub(exc)}


def read(path):
    try:
        return scrub(Path(path).read_text(errors="replace"))
    except OSError as exc:
        return {"error": str(exc)}


HASH_CACHE = {}


def file_info(path, hash_file=True):
    path = Path(path)
    try:
        stat = path.stat()
        result = {"realpath": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if hash_file:
            key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            if key not in HASH_CACHE:
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(8 << 20), b""):
                        h.update(block)
                HASH_CACHE[key] = h.hexdigest()
            result["sha256"] = HASH_CACHE[key]
        return result
    except OSError as exc:
        return {"error": str(exc)}


def walk(root, limit=100000):
    """Bound discovery and expose truncation instead of silently implying exhaustive coverage."""
    paths, errors = [], []
    for directory, dirs, files in os.walk(root, onerror=lambda e: errors.append(str(e))):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__", ".cache", "node_modules"))
        for name in sorted(files):
            paths.append(Path(directory) / name)
            if len(paths) >= limit:
                return paths, {"truncated": True, "limit": limit, "errors": errors}
    return paths, {"truncated": False, "errors": errors}


def manifest(root, predicate=lambda p: True, hash_file=True):
    files, coverage = walk(root)
    return {
        "root": str(root),
        "coverage": coverage,
        "files": {str(p.relative_to(root)): file_info(p, hash_file) for p in files if predicate(p)},
    }


def git_info(directory):
    top = command(["git", "-C", str(directory), "rev-parse", "--show-toplevel"])
    if top.get("returncode") != 0:
        return {"available": False, "error": top}
    root = top["stdout"].strip()
    diff = subprocess.run(["git", "-C", root, "diff", "HEAD", "--binary"], capture_output=True)
    untracked = command(["git", "-C", root, "ls-files", "--others", "--exclude-standard"])
    return {
        "root": root,
        "head": command(["git", "-C", root, "rev-parse", "HEAD"]),
        "status": command(["git", "-C", root, "status", "--porcelain"]),
        "diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None,
        "diff_returncode": diff.returncode,
        "untracked": untracked,
    }


def loaded_paths():
    maps = read("/proc/self/maps")
    if not isinstance(maps, str):
        return []
    return sorted(
        {
            line.split(None, 5)[5].removesuffix(" (deleted)")
            for line in maps.splitlines()
            if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith("/")
        }
    )


def probe(output):
    """Separate disposable process; no tensor operations, collectives or synchronize calls."""
    report = {"python": sys.executable, "sys_path": sys.path, "modules": {}, "device": {}}
    for name in MODULES:
        print(f"[import probe] {name}", flush=True)
        try:
            m = importlib.import_module(name)
            report["modules"][name] = {
                "origin": getattr(m, "__file__", None),
                "paths": list(getattr(m, "__path__", [])),
                "version": str(getattr(m, "__version__", "unknown")),
                "git_version": str(getattr(getattr(m, "version", None), "git_version", "unknown")),
            }
            if name == "torch":
                report["torch_config"] = m.__config__.show()
                report["torch_parallel"] = m.__config__.parallel_info()
            if name == "torch_npu":
                import torch

                count = torch.npu.device_count()
                report["device"]["count"] = count
                for i in range(count):
                    try:
                        report["device"][str(i)] = {
                            "name": torch.npu.get_device_name(i),
                            "properties": str(torch.npu.get_device_properties(i)),
                        }
                    except Exception as exc:
                        report["device"][str(i)] = {"error": repr(exc)}
            if name == "acl" and hasattr(m, "get_version"):
                report["acl_version"] = m.get_version()
        except Exception:
            report["modules"].setdefault(name, {})["error"] = traceback.format_exc()
            try:
                spec = importlib.util.find_spec(name)
                if spec:
                    report["modules"][name].update(
                        origin=spec.origin, paths=list(spec.submodule_search_locations or [])
                    )
            except Exception:
                pass
        report["loaded_paths"] = loaded_paths()
        dump(output, report)  # Retain the last completed module even if the next import hangs/crashes.
    torch = sys.modules.get("torch")
    if torch is not None:
        report["precision_flags"] = {}
        for key in (
            "npu.config.allow_internal_format",
            "npu.conv.allow_hf32",
            "npu.matmul.allow_hf32",
            "backends.cuda.matmul.allow_tf32",
            "backends.cudnn.allow_tf32",
        ):
            try:
                value = torch
                for component in key.split("."):
                    value = getattr(value, component)
                report["precision_flags"][key] = str(value)
            except Exception as exc:
                report["precision_flags"][key] = {"error": str(exc)}
        for name in ("get_float32_matmul_precision", "are_deterministic_algorithms_enabled", "get_num_threads"):
            try:
                report["precision_flags"][name] = getattr(torch, name)()
            except Exception as exc:
                report["precision_flags"][name] = {"error": str(exc)}
    dump(output, report)


def collect(args):
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "label": args.label,
        "image_digest": args.image_digest,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "python_realpath": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "sys_path": sys.path,
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "collector_sha256": file_info(__file__)["sha256"],
        "environment": {
            k: scrub(v)
            for k, v in sorted(os.environ.items())
            if (k.startswith(PREFIXES) or k in EXACT) and not SECRET.search(k)
        },
        "ci": {
            k: os.environ.get(k)
            for k in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "RUNNER_NAME", "RUNNER_ARCH", "RUNNER_OS")
        },
    }
    dump(out / "report.json", report)
    print("[1/6] OS / driver / topology / limits", flush=True)
    cmds = {
        "uname": ["uname", "-a"],
        "lscpu": ["lscpu"],
        "numa": ["numactl", "--hardware"],
        "npu_info": ["npu-smi", "info"],
        "npu_topology": ["npu-smi", "info", "-t", "topo"],
        "shm": ["df", "-h", "/dev/shm"],
        "disk": ["df", "-h"],
        "mounts": ["findmnt", "-rn", "-o", "TARGET,SOURCE,FSTYPE"],
        "ldconfig": ["ldconfig", "-p"],
        "glibc": ["getconf", "GNU_LIBC_VERSION"],
        "kernel_modules": ["lsmod"],
        "os_packages_deb": ["dpkg-query", "-W", "-f=${Package}\t${Version}\n"],
        "os_packages_rpm": ["rpm", "-qa"],
        "processes": ["ps", "-eo", "pid,ppid,comm,rss,etime"],
    }
    report["commands"] = {name: command(argv) for name, argv in cmds.items()}
    files = [
        "/etc/os-release",
        "/proc/meminfo",
        "/proc/cpuinfo",
        "/proc/self/limits",
        "/proc/self/status",
        "/proc/self/cgroup",
        "/sys/kernel/mm/transparent_hugepage/enabled",
        "/proc/sys/vm/max_map_count",
        "/proc/sys/vm/overcommit_memory",
        "/proc/sys/kernel/numa_balancing",
        "/etc/ascend_install.info",
        "/usr/local/Ascend/driver/version.info",
        "/usr/local/Ascend/firmware/version.info",
    ]
    cg_roots = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory"), Path("/sys/fs/cgroup/cpuset")]
    cgroup = read("/proc/self/cgroup")
    if isinstance(cgroup, str):
        for line in cgroup.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3:
                candidate = (Path("/sys/fs/cgroup") / parts[2].lstrip("/")).resolve()
                if str(candidate).startswith("/sys/fs/cgroup/"):
                    cg_roots.append(candidate)
    for root in cg_roots:
        files.extend(
            str(root / name)
            for name in (
                "memory.max",
                "memory.current",
                "memory.events",
                "cpu.max",
                "pids.max",
                "pids.current",
                "cpuset.cpus.effective",
                "cpuset.mems.effective",
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
                "cpuset.cpus",
                "cpuset.mems",
            )
        )
    files += [str(p) for p in Path("/sys/devices/system/node").glob("node*/meminfo")]
    report["system_files"] = {p: read(p) for p in files}
    report["packages"] = {}
    for d in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", d.metadata.get("Name", "unknown")).lower()
        report["packages"][name] = {
            "version": d.version,
            "location": str(d.locate_file("")),
            "direct_url": scrub(d.read_text("direct_url.json") or ""),
            "installer": d.read_text("INSTALLER"),
        }
    dump(out / "report.json", report)
    print("[2/6] isolated imports and loaded library paths", flush=True)
    report["probe_process"] = command(
        [sys.executable, str(Path(__file__).resolve()), "--probe-output", str(out / "probe.json")],
        timeout=args.import_timeout,
    )
    probe_report = json.loads((out / "probe.json").read_text()) if (out / "probe.json").exists() else {}
    report["probe"] = probe_report
    report["collection_issues"] = [
        f"import {name}: failed or not reached"
        for name in MODULES
        if name not in probe_report.get("modules", {}) or probe_report["modules"][name].get("error")
    ]
    print("[3/6] installed Python source and native binary hashes", flush=True)
    report["module_files"] = {}
    report["git"] = {}
    for name, module in probe_report.get("modules", {}).items():
        origin = module.get("origin")
        if not origin:
            continue
        print(f"  hashing {name}", flush=True)
        root = Path(origin).parent
        if module.get("paths"):
            report["module_files"][name] = manifest(
                root, lambda p: p.suffix in (".py", ".json", ".yaml", ".yml") or ".so" in p.name
            )
        else:
            report["module_files"][name] = {"root": str(root), "files": {Path(origin).name: file_info(origin)}}
        report["git"][name] = git_info(root)
    if args.repo:
        report["git"]["requested_repo"] = git_info(args.repo)
    report["loaded_libraries"] = {p: file_info(p) for p in probe_report.get("loaded_paths", []) if ".so" in p}
    dump(out / "report.json", report)
    print("[4/6] CANN/OPP/driver manifests and key libraries", flush=True)
    roots = [
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
        Path("/usr/local/Ascend/driver"),
        Path("/usr/local/Ascend/firmware"),
        Path("/usr/local/Ascend/nnal"),
    ]
    for key in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH", "ASCEND_CUSTOM_OPP_PATH"):
        roots.extend(Path(p) for p in os.environ.get(key, "").split(os.pathsep) if p)
    roots.extend(args.cann_root)
    report["cann"] = {}
    seen = set()
    patterns = (
        "*version*.info",
        "*version*.cfg",
        "*version*.json",
        "*install.info",
        "version.txt",
        "libascendcl.so*",
        "libhccl.so*",
        "libacl_op_compiler.so*",
        "libopapi.so*",
        "libnnopbase.so*",
        "libascend_hal.so*",
        "libcann*.so*",
        "libruntime.so*",
        "libacl*.so*",
    )
    for root in roots:
        real = root.resolve()
        if real in seen:
            continue
        seen.add(real)
        if not root.exists():
            report["cann"][str(root)] = {"missing": True}
            continue
        m = manifest(root, lambda p: any(fnmatch.fnmatch(p.name, pat) for pat in patterns))
        for rel, info in m["files"].items():
            if ".so" not in Path(rel).name and info.get("size", 0) < 256000:
                info["content"] = read(root / rel)
        report["cann"][str(root)] = m
    print("[5/6] model/data/config fingerprints", flush=True)
    report["assets"] = {}
    for path in args.asset:
        path = path.expanduser().resolve()
        report["assets"][str(path)] = manifest(path) if path.is_dir() else file_info(path)
    for section in ("module_files", "cann", "assets"):
        for key, value in report[section].items():
            if value.get("coverage", {}).get("truncated"):
                report["collection_issues"].append(f"{section}/{key}: file discovery truncated")
    dump(out / "report.json", report)
    print("[6/6] finished", flush=True)
    summary = {
        "label": args.label,
        "image_digest": args.image_digest,
        "python": sys.executable,
        "device": probe_report.get("device"),
        "modules": probe_report.get("modules"),
        "probe_returncode": report["probe_process"].get("returncode"),
        "collection_issues": report["collection_issues"],
        "report": str(out / "report.json"),
    }
    dump(out / "summary.json", summary)
    print(scrub(json.dumps(summary, ensure_ascii=False, indent=2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="local")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--asset", type=Path, action="append", default=[], help="完整哈希模型/数据/配置；可重复")
    parser.add_argument("--cann-root", type=Path, action="append", default=[])
    parser.add_argument("--image-digest", default=os.environ.get("REPRO_IMAGE_DIGEST", "unknown"))
    parser.add_argument("--import-timeout", type=int, default=180)
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_output:
        probe(args.probe_output)
    elif args.output:
        collect(args)
    else:
        parser.error("--output is required")


if __name__ == "__main__":
    main()
