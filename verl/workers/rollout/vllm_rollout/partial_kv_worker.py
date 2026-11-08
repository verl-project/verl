# Copyright 2026 Individual Contributor: HaitaoWuTJU
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
"""CuMem worker extension for in-process full-pool partial-rollout offload."""

from vllm.device_allocator import get_mem_allocator_instance
from vllm.device_allocator.sleep_mode_backend import CuMemBackend

from verl.workers.rollout.vllm_rollout.partial_kv import PROTOCOL
from verl.workers.rollout.vllm_rollout.partial_kv_memory import (
    RUNTIME_TAG,
    admit_cpu_copy,
    protect_runtime_state,
    runtime_tensors,
)
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


def pool_stats(allocator):
    """Report allocated, unmapped and CPU-backup physical bytes for every tag."""
    pools = {}
    for data in allocator.pointer_to_data.values():
        row = pools.setdefault(data.tag, dict(allocated_bytes=0, asleep_bytes=0, cpu_backup_bytes=0))
        size = int(data.handle[1])
        row["allocated_bytes"] += size
        row["asleep_bytes"] += size if data.is_asleep else 0
        if data.cpu_backup_tensor is not None:
            backup = data.cpu_backup_tensor
            row["cpu_backup_bytes"] += backup.numel() * backup.element_size()
    return pools


class PartialKVWorkerExtension(vLLMColocateWorkerExtension):
    """Keep native buffer, communication and weight-refit behavior intact."""

    def partial_kv_stats(self):
        """Return physical pool accounting without reading GPU contents."""
        import os
        from pathlib import Path

        from verl.utils.device import get_torch_device

        device = get_torch_device()
        index = device.current_device()
        return dict(
            protocol=PROTOCOL,
            pools=pool_stats(get_mem_allocator_instance()),
            pid=os.getpid(),
            cgroup=Path("/proc/self/cgroup").read_text(),
            device=index,
            uuid=str(device.get_device_properties(index).uuid),
        )

    def partial_kv_hashes(self):
        """Expensive validation RPC; hash mapped KV and live state, never hot-path data."""
        import hashlib

        import torch
        from vllm.device_allocator.cumem import libcudart

        torch.accelerator.synchronize()
        digest = hashlib.sha256()
        for pointer, allocation in sorted(get_mem_allocator_instance().pointer_to_data.items()):
            if allocation.tag != "kv_cache":
                continue
            if allocation.is_asleep:
                raise RuntimeError("Cannot hash unmapped partial KV")
            buffer = torch.empty(allocation.handle[1], dtype=torch.uint8, device="cpu", pin_memory=True)
            libcudart.cudaMemcpy(buffer.data_ptr(), pointer, allocation.handle[1])
            digest.update(int(pointer).to_bytes(8, "little"))
            digest.update(buffer.numpy().tobytes())
        state = {}
        for path, tensor in runtime_tensors(self.model_runner):
            payload = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            state[path] = hashlib.sha256(payload).hexdigest()
        return dict(kv_cache=digest.hexdigest(), runtime_state=state)

    def offload_partial_kv(self, preserve_kv, headroom_gb):
        """Preserve every non-weight allocation at its original virtual address."""
        if type(preserve_kv) is not bool or type(headroom_gb) not in (int, float) or headroom_gb < 0:
            raise ValueError("Invalid partial KV preservation policy")
        allocator = get_mem_allocator_instance()
        getter = getattr(self, "_get_sleep_mode_backend", None)
        backend = getter() if getter is not None else self.sleep_mode_backend
        if type(backend) is not CuMemBackend:
            raise RuntimeError("Partial KV preservation requires the inspected CuMem backend")
        self._partial_kv_backup = None
        runtime = protect_runtime_state(self.model_runner, allocator)
        allocations = list(allocator.pointer_to_data.values())
        if not any(data.tag == "kv_cache" for data in allocations) or any(data.is_asleep for data in allocations):
            raise RuntimeError("Cannot preserve missing or partially sleeping KV pools")
        excluded = {"weights"} if preserve_kv else {"weights", "kv_cache"}
        retained = tuple(sorted({data.tag for data in allocations} - excluded))
        required = sum(int(data.handle[1]) for data in allocations if data.tag in retained)
        original = backend.suspend

        def suspend(level):
            if level != 2:
                raise RuntimeError("Partial KV handoff requires native level-2 sleep")
            backend._state = "SUSPENDED"
            with admit_cpu_copy(required, int(headroom_gb * 1024**3)):
                allocator.sleep(offload_tags=retained)

        backend.suspend = suspend
        try:
            # Worker.sleep keeps model buffers/special parameters and native
            # synchronization. Engine-level sleep would reset the scheduler KV.
            self.sleep(level=2)
        finally:
            backend.suspend = original
        pools = pool_stats(allocator)
        if any(
            row["asleep_bytes"] != row["allocated_bytes"]
            or row["cpu_backup_bytes"] != (row["allocated_bytes"] if tag in retained else 0)
            for tag, row in pools.items()
        ):
            raise RuntimeError("Full-pool CPU backup is incomplete")
        self._partial_kv_backup = dict(preserve=preserve_kv, allocations=allocations)
        return dict(protocol=PROTOCOL, kv_preserved=preserve_kv, asleep=True, pools=pools, runtime_state=runtime)

    def restore_partial_kv(self, tags):
        """Restore native pools and verify runtime and KV allocation readiness."""
        if not isinstance(tags, list) or not tags or any(tag not in ("weights", "kv_cache") for tag in tags):
            raise ValueError("Invalid partial KV wake tags")
        backup = getattr(self, "_partial_kv_backup", None)
        if backup is None:
            raise RuntimeError("CPU KV restore has no acknowledged backup")
        self.wake_up(tags=tags)
        allocator = get_mem_allocator_instance()
        allocations = backup["allocations"]
        runtime = [data for data in allocations if data.tag == RUNTIME_TAG]
        if any(data.is_asleep or data.cpu_backup_tensor is not None for data in runtime) or any(
            data.is_asleep or data.cpu_backup_tensor is not None
            for data in allocations
            if data.tag not in ("weights", "kv_cache")
        ):
            raise RuntimeError("Live request state was not completely restored")
        if "kv_cache" in tags and any(
            data.is_asleep or data.cpu_backup_tensor is not None for data in allocations if data.tag == "kv_cache"
        ):
            raise RuntimeError("KV pool was not completely restored")
        if "weights" in tags and any(data.is_asleep for data in allocations if data.tag == "weights"):
            raise RuntimeError("Weight allocations were not completely restored")
        if any(data.is_asleep for data in allocator.pointer_to_data.values()):
            getter = getattr(self, "_get_sleep_mode_backend", None)
            backend = getter() if getter is not None else self.sleep_mode_backend
            backend._state = "SUSPENDED"
        return dict(
            protocol=PROTOCOL,
            tags=tags,
            kv_preserved=backup["preserve"],
            runtime_restored=True,
            pools=pool_stats(allocator),
        )
