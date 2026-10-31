# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Copy a HF checkpoint from a shared (GCS Fuse) path to the same local path on every alive TPU Ray node.

TorchTitan's HF reader and vLLM mmap the safetensors shards. Over GCS Fuse those page faults can be very slow
(~20 MB/s per host), while a parallel sequential copy runs at ~1 GB/s. Spot TPU slices are recreated with an empty
host /tmp, so this runs before every job and skips nodes whose copy is already complete (same file names and sizes).

Usage: python3 stage_model_local.py <src_dir> <dst_dir> [parallel_files]
"""

import os
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _manifest(path: str) -> dict[str, int]:
    return {
        f: os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if not f.startswith(".") and os.path.isfile(os.path.join(path, f))
    }


_remote_manifest = ray.remote(num_cpus=0)(_manifest)


@ray.remote(num_cpus=0)
def _stage(src: str, dst: str, parallel: int, expected: dict[str, int], required: bool) -> str:
    import socket
    import subprocess

    host = socket.gethostname()
    os.makedirs(dst, exist_ok=True)
    have = _manifest(dst)
    missing = [f for f, size in expected.items() if have.get(f) != size]
    if not missing:
        return f"{host}: already staged ({len(expected)} files)"
    if not os.path.isdir(src):
        if required:
            raise FileNotFoundError(f"{host}: {src} does not exist on this node")
        return f"{host}: {src} is not mounted here, skipped"
    t0 = time.time()
    procs = []
    for f in missing:
        while len(procs) >= parallel:
            procs = [p for p in procs if p.poll() is None]
            time.sleep(0.2)
        procs.append(subprocess.Popen(["cp", os.path.join(src, f), os.path.join(dst, f)]))
    for p in procs:
        p.wait()
    have = _manifest(dst)
    failed = [f for f, size in expected.items() if have.get(f) != size]
    if failed:
        raise RuntimeError(f"{host}: staging incomplete, mismatched files: {failed}")
    gb = sum(expected[f] for f in missing) / 1e9
    dt = time.time() - t0
    return f"{host}: copied {len(missing)} files ({gb:.1f} GB) in {dt:.0f}s ({gb / max(dt, 1e-3):.2f} GB/s)"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    parallel = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    if os.path.abspath(src) == os.path.abspath(dst):
        print(f"[stage_model_local] src == dst ({src}); nothing to do")
        return
    ray.init(address="auto", logging_level="ERROR")
    alive = [n for n in ray.nodes() if n.get("Alive")]
    tpu_nodes = [n for n in alive if n.get("Resources", {}).get("TPU", 0) > 0]
    cpu_nodes = [n for n in alive if n.get("Resources", {}).get("TPU", 0) == 0]
    if not tpu_nodes:
        raise RuntimeError("no alive TPU nodes in the Ray cluster")

    def on(node):
        return NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)

    # List the checkpoint on a TPU node: the driver (e.g. on a CPU head) may not mount the shared path.
    expected = ray.get(_remote_manifest.options(scheduling_strategy=on(tpu_nodes[0])).remote(src))
    if not any(f.endswith(".safetensors") for f in expected):
        raise FileNotFoundError(f"no safetensors found in {src}")
    # CPU-only nodes (e.g. the Ray head, where the TaskRunner builds HFModelConfig / the tokenizer from MODEL_PATH)
    # only need the config, tokenizer and index files, not the weights; they are skipped if they do not mount src.
    light = {f: size for f, size in expected.items() if not f.endswith((".safetensors", ".bin", ".pt"))}
    print(
        f"[stage_model_local] {src} -> {dst}: full copy on {len(tpu_nodes)} TPU nodes "
        f"({sum(expected.values()) / 1e9:.1f} GB), config/tokenizer only on {len(cpu_nodes)} CPU nodes"
    )
    refs = [
        _stage.options(scheduling_strategy=on(n)).remote(src, dst, parallel, files, required)
        for nodes, files, required in ((tpu_nodes, expected, True), (cpu_nodes, light, False))
        for n in nodes
    ]
    t0 = time.time()
    for line in ray.get(refs):
        print(f"[stage_model_local] {line}")
    print(f"[stage_model_local] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
