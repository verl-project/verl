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
"""
Multi-process distributed tests for Vision DP gather + backward.

Uses NCCL backend on CUDA (one GPU per worker via CUDA_VISIBLE_DEVICES).
Validates that empty-rank backward (P1 bug fix) does not hang and
that gradients flow correctly through GatherVisionEmbeddings.
"""

import os
import sys
import subprocess

import pytest
import torch

from verl.utils.vision_dp import assign_images_to_dp_ranks


# ---------------------------------------------------------------------------
# Worker entry point (run as a separate script via subprocess)
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = r"""
import os
import sys
import torch
import torch.distributed as dist

# Setup
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
patch_sizes = list(map(int, os.environ["PATCH_SIZES"].split(",")))
hidden_size = 64

dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
    rank=rank,
    world_size=world_size,
)
torch.cuda.set_device(0)  # each worker sees exactly 1 GPU via CUDA_VISIBLE_DEVICES
device = torch.device("cuda", 0)

dp_group = dist.group.WORLD

sys.path.insert(0, os.environ.get("PROJECT_ROOT", "."))
from verl.utils.vision_dp import assign_images_to_dp_ranks, GatherVisionEmbeddings

image_assignments, _ = assign_images_to_dp_ranks(patch_sizes, world_size)
local_indices = image_assignments[rank]
local_count = sum(patch_sizes[i] for i in local_indices)

if local_count > 0:
    local_emb = torch.randn(local_count, hidden_size, device=device, requires_grad=True)
else:
    local_emb = torch.zeros(0, hidden_size, device=device, requires_grad=True)

all_counts = [
    sum(patch_sizes[i] for i in image_assignments[r])
    for r in range(world_size)
]

# Forward
result = GatherVisionEmbeddings.apply(local_emb, dp_group, all_counts)

total_expected = sum(patch_sizes)
assert result.shape == (total_expected, hidden_size), (
    f"rank={rank}: result.shape={result.shape}, expected=({total_expected}, {hidden_size})"
)

# Backward — if empty rank doesn't participate, this hangs
loss = result.sum()
loss.backward()

assert local_emb.grad is not None, f"rank={rank}: local_emb.grad is None"
assert local_emb.grad.shape == (local_count, hidden_size), (
    f"rank={rank}: grad.shape={local_emb.grad.shape}, expected=({local_count}, {hidden_size})"
)

dist.destroy_process_group()
print(f"rank={rank} OK")
"""


def _gpu_count():
    """Return number of visible CUDA GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _run_distributed_test(world_size, tmp_path, patch_sizes):
    """Launch distributed workers as subprocesses, each pinned to one GPU."""
    available_gpus = _gpu_count()
    if available_gpus < world_size:
        pytest.skip(f"Need {world_size} GPUs, only {available_gpus} available")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Find a free port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        master_port = str(s.getsockname()[1])

    env = os.environ.copy()
    env["WORLD_SIZE"] = str(world_size)
    env["MASTER_PORT"] = master_port
    env["PATCH_SIZES"] = ",".join(str(s) for s in patch_sizes)
    env["PROJECT_ROOT"] = project_root

    script_path = str(tmp_path / "worker.py")
    with open(script_path, "w") as f:
        f.write(_WORKER_SCRIPT)

    procs = []
    for rank in range(world_size):
        rank_env = env.copy()
        rank_env["RANK"] = str(rank)
        rank_env["CUDA_VISIBLE_DEVICES"] = str(rank)
        p = subprocess.Popen(
            [sys.executable, script_path],
            env=rank_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root,
        )
        procs.append(p)

    for p in procs:
        p.wait(timeout=60)

    for rank, p in enumerate(procs):
        stdout = p.stdout.read().decode()
        stderr = p.stderr.read().decode()
        if p.returncode != 0:
            pytest.fail(
                f"Worker rank={rank} failed (exit={p.returncode}).\n"
                f"stdout: {stdout}\nstderr: {stderr}"
            )
        assert f"rank={rank} OK" in stdout, (
            f"rank={rank} did not print OK.\nstdout: {stdout}\nstderr: {stderr}"
        )


class TestVisionDPDistributed:
    """Distributed tests using NCCL backend on CUDA."""

    def test_normal_4_images_2_ranks(self, tmp_path):
        """Normal case: 4 images across 2 ranks, no empty ranks."""
        _run_distributed_test(2, tmp_path, [100, 100, 100, 100])

    def test_critical_2_images_2_ranks(self, tmp_path):
        """Critical: each rank gets exactly 1 image."""
        _run_distributed_test(2, tmp_path, [100, 200])

    def test_p1_empty_rank_2_images_4_ranks(self, tmp_path):
        """P1 bug: 2 images, 4 ranks -> 2 ranks empty, backward must not hang."""
        _run_distributed_test(4, tmp_path, [100, 200])

    def test_extreme_1_image_4_ranks(self, tmp_path):
        """Extreme: 1 image, 4 ranks -> 3 ranks empty."""
        _run_distributed_test(4, tmp_path, [300])

    def test_unbalanced_load(self, tmp_path):
        """Unbalanced: one huge image among small ones."""
        _run_distributed_test(2, tmp_path, [4096, 256, 256, 256])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
