# Copyright 2026 Alibaba Group
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
#
# Adapted from SkyRL (Apache 2.0):
#   https://github.com/NovaSky-AI/SkyRL/blob/main/examples/train_integrations/harbor/prepare_harbor_dataset.py
"""
Prepare Harbor task datasets from HuggingFace Hub.

Downloads a dataset and extracts Harbor tasks from parquet files containing
tar-archived task directories (columns: path, task_binary).

Output directory defaults to ~/data/harbor/<repo-name>.

Usage:

    python verl/experimental/agentic_rl_harbor/prepare_harbor_dataset.py \
        --dataset DCAgent/exp_rpt_curriculum-easy
"""

import argparse
import io
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq


def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(base.resolve())
    except Exception:
        return False


def _sanitize_tar_member_name(name: str) -> str:
    p = PurePosixPath(name)
    parts = [part for part in p.parts if part not in ("..", ".", "")]
    while parts and parts[0] == "/":
        parts.pop(0)
    return str(PurePosixPath(*parts)) if parts else ""


def _safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            member_name = _sanitize_tar_member_name(member.name)
            if not member_name or member_name.endswith("/"):
                (dest_dir / member_name).mkdir(parents=True, exist_ok=True)
                continue
            if ".snapshot" in PurePosixPath(member_name).parts:
                continue
            target = (dest_dir / member_name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isfile():
                with tf.extractfile(member) as src:
                    if src is None:
                        continue
                    with open(target, "wb") as dst:
                        dst.write(src.read())
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)


def _extract_one(args: tuple) -> bool:
    rel_path, data, output_dir_str = args
    if not isinstance(rel_path, str) or not isinstance(data, bytes | bytearray | memoryview):
        return False
    output_dir = Path(output_dir_str)

    safe_rel = PurePosixPath(rel_path)
    parts = [p for p in safe_rel.parts if p not in ("..", "")]
    rel_norm = Path(*parts) if parts else Path("task_unknown")
    target_dir = (output_dir / rel_norm).resolve()

    if not _is_within(output_dir, target_dir):
        return False

    if target_dir.exists() and (target_dir / "instruction.md").exists():
        return True

    try:
        _safe_extract_tar(bytes(data), target_dir)
        return True
    except Exception as e:
        print(f"  Warning: Failed to extract {rel_path}: {e}")
        return False


def extract_parquet(parquet_path: Path, output_dir: Path) -> int:
    table = pq.read_table(parquet_path)
    path_col = table.column("path").to_pylist()
    data_col = table.column("task_binary").to_pylist()

    output_dir.mkdir(parents=True, exist_ok=True)
    args = [(p, d, str(output_dir)) for p, d in zip(path_col, data_col, strict=False)]

    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_extract_one, args, chunksize=64))

    return sum(results)


def prepare(dataset_name: str, output_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    repo_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    if output_dir is None:
        output_dir = os.path.join("~/data/harbor", repo_name)
    output_path = Path(os.path.expanduser(output_dir)).resolve()

    print(f"Downloading {dataset_name}...")
    snapshot_dir = Path(snapshot_download(repo_id=dataset_name, repo_type="dataset"))
    print(f"Downloaded to {snapshot_dir}")

    parquets = []
    for f in snapshot_dir.glob("**/*.parquet"):
        try:
            schema = pq.read_schema(f)
            if "path" in schema.names and "task_binary" in schema.names:
                parquets.append(f)
        except Exception as e:
            print(f"  Warning: Could not read schema from {f}: {e}")
            continue

    if not parquets:
        print("No parquet files found, symlinking snapshot directly...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_symlink():
            output_path.unlink()
        elif output_path.exists():
            shutil.rmtree(output_path)
        output_path.symlink_to(snapshot_dir)
        print(f"Done! Symlinked {output_path} -> {snapshot_dir}")
        return str(output_path)

    total = 0
    for pq_file in parquets:
        print(f"Extracting {pq_file.name}...")
        total += extract_parquet(pq_file, output_path)

    print(f"Done! {total} tasks extracted to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Harbor task dataset from HuggingFace Hub")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset (e.g. DCAgent/exp_rpt_curriculum-easy)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: ~/data/harbor/<repo-name>)")
    args = parser.parse_args()
    prepare(args.dataset, args.output_dir)
