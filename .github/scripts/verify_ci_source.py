"""Bind a shared CI source bundle and every Ray node to the checked-out commit."""

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

MANIFEST = ".verl-ci-source.json"
IMPORTANT_FILES = (
    ".github/scripts/run_multinode_hccl_diagnostics.py",
    ".github/scripts/verify_ci_source.py",
    "verl/utils/vllm/npu_vllm_patch.py",
    "verl/utils/hccl_diagnostics.py",
    "verl/workers/engine/megatron/transformer_impl.py",
    "tests/special_npu/nightly_ci_ascend/run_grpo_glm52_top16_megatron_npu_multinode.sh",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(repo, commit):
    files = {}
    for path in sorted(repo.rglob("*")):
        name = path.relative_to(repo).as_posix()
        if name == MANIFEST:
            continue
        if path.is_symlink():
            files[name] = {"symlink": os.readlink(path)}
        elif path.is_file():
            files[name] = {"sha256": digest(path)}
    if not all(name in files for name in IMPORTANT_FILES):
        raise RuntimeError("The source bundle is missing diagnostic or workload files")
    (repo / MANIFEST).write_text(json.dumps({"commit": commit, "files": files}, sort_keys=True) + "\n")


def verify_source(repo, commit, manifest_sha256, check_import=False):
    manifest_path = repo / MANIFEST
    if digest(manifest_path) != manifest_sha256:
        raise RuntimeError(f"Manifest mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["commit"] != commit:
        raise RuntimeError(f"Commit mismatch: expected {commit}, found {manifest['commit']}")
    for name, expected in manifest["files"].items():
        path = repo / name
        if "symlink" in expected:
            matches = path.is_symlink() and os.readlink(path) == expected["symlink"]
        else:
            matches = path.is_file() and not path.is_symlink() and digest(path) == expected["sha256"]
        if not matches:
            raise RuntimeError(f"Source mismatch: {path}")
    record = {
        "hostname": socket.gethostname(),
        "repo": str(repo),
        "commit": commit,
        "manifest_sha256": manifest_sha256,
        "verified_files": len(manifest["files"]),
        "python": sys.executable,
        "sha256": {name: manifest["files"][name]["sha256"] for name in IMPORTANT_FILES},
    }
    if check_import:
        spec = importlib.util.find_spec("verl")
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin != (repo / "verl/__init__.py").resolve():
            raise RuntimeError(f"verl import mismatch: expected {repo}/verl/__init__.py, found {origin}")
        record["verl_import"] = str(origin)
    print("[VERL_HCCL_SOURCE] " + json.dumps(record, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--write", metavar="COMMIT")
    parser.add_argument("--check-import", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.write:
        write_manifest(repo, args.write)
    else:
        try:
            verify_source(
                repo,
                os.environ["VERL_DIAGNOSTIC_COMMIT"],
                os.environ["VERL_HCCL_SOURCE_MANIFEST_SHA256"],
                args.check_import,
            )
        except Exception as error:
            print(f"[VERL_HCCL_SOURCE] FAILED hostname={socket.gethostname()} repo={repo}: {error}", flush=True)
            raise


if __name__ == "__main__":
    main()
