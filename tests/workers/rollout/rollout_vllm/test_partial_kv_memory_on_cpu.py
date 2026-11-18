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
"""CPU tests of actual pool policy, mixed ownership and copy admission."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load(name):
    path = Path(__file__).resolve().parents[4] / "verl/workers/rollout/vllm_rollout" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


memory = _load("partial_kv_memory")
kv = _load("partial_kv")


class _Tensor:
    """A CUDA tensor view, used only for address ownership classification."""

    is_cuda = True

    def __init__(self, pointer):
        self.pointer = pointer

    def numel(self):
        return 1

    def data_ptr(self):
        return self.pointer


class _Backup:
    def __init__(self, contents):
        self.contents = bytes(contents)

    def numel(self):
        return len(self.contents)

    def element_size(self):
        return 1


class _Allocator:
    """Model CuMem mapping semantics and poison discarded physical contents."""

    def __init__(self):
        self.pointer_to_data = {}
        self.copies = 0
        for pointer, size, tag in (
            (1000, 100, "weights"),
            (2000, 128, "kv_cache"),
            (3000, 64, "weights"),
            (4000, 32, "runtime"),
            (5000, 16, "future_workspace_tag"),
        ):
            self.pointer_to_data[pointer] = SimpleNamespace(
                handle=(pointer, size),
                tag=tag,
                is_asleep=False,
                cpu_backup_tensor=None,
                contents=bytes([pointer // 1000]) * size,
            )

    def sleep(self, offload_tags):
        for row in self.pointer_to_data.values():
            if row.tag in offload_tags:
                row.cpu_backup_tensor = _Backup(row.contents)
            row.contents = b"\xa5" * row.handle[1]
            row.is_asleep = True
        self.copies += 1

    def wake_up(self, tags):
        for row in self.pointer_to_data.values():
            if row.is_asleep and (tags is None or row.tag in tags or row.tag not in ("weights", "kv_cache")):
                row.is_asleep = False
                if row.cpu_backup_tensor is not None:
                    row.contents = row.cpu_backup_tensor.contents
                    row.cpu_backup_tensor = None


@pytest.fixture
def worker_fixture(monkeypatch, tmp_path):
    allocator = _Allocator()

    class Backend:
        def suspend(self, level):
            raise AssertionError("Worker must install its explicit full-pool policy")

        def resume(self, tags):
            allocator.wake_up(tags)
            self._state = "RUNNING"

    class NativeWorker:
        def sleep(self, level):
            self.buffer_backup = self.model_buffer
            self.sleep_mode_backend.suspend(level)
            self.model_buffer = "released"

        def wake_up(self, tags):
            self.sleep_mode_backend.resume(tags)
            if "weights" in tags:
                self.model_buffer = self.buffer_backup

    modules = {
        "vllm.device_allocator": dict(get_mem_allocator_instance=lambda: allocator),
        "vllm.device_allocator.sleep_mode_backend": dict(CuMemBackend=Backend),
        "verl.workers.rollout.vllm_rollout.utils": dict(vLLMColocateWorkerExtension=NativeWorker),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.partial_kv", kv)
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.partial_kv_memory", memory)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(is_tensor=lambda value: isinstance(value, _Tensor)))
    monkeypatch.setattr(memory, "host_memory_available", lambda: 10**12)
    original_admit = memory.admit_cpu_copy
    monkeypatch.setattr(
        memory, "admit_cpu_copy", lambda required, headroom: original_admit(required, headroom, directory=tmp_path)
    )
    module = _load("partial_kv_worker")
    worker = module.PartialKVWorkerExtension()
    worker.sleep_mode_backend = Backend()
    worker.model_buffer = "original buffer"
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(parameter=_Tensor(1012)),
        model_state=SimpleNamespace(index=_Tensor(3012)),
        sampler=SimpleNamespace(count=_Tensor(3020)),
    )
    return worker, allocator, module


def test_mixed_allocations_and_unknown_pools_survive_full_roundtrip(worker_fixture):
    worker, allocator, _ = worker_fixture
    before = {ptr: row.contents for ptr, row in allocator.pointer_to_data.items()}
    receipt = worker.offload_partial_kv(True, 0)
    assert receipt["kv_preserved"] and receipt["asleep"]
    assert receipt["runtime_state"]["protected_allocation_bytes"] == 64
    assert sum(row["cpu_backup_bytes"] for row in receipt["pools"].values()) == 240
    assert receipt["pools"]["weights"]["cpu_backup_bytes"] == 0
    assert receipt["pools"]["kv_cache"]["cpu_backup_bytes"] == 128
    assert all(row.is_asleep for row in allocator.pointer_to_data.values())
    assert allocator.pointer_to_data[3000].tag == memory.RUNTIME_TAG
    worker.restore_partial_kv(["weights"])
    assert worker.model_buffer == "original buffer"
    assert allocator.pointer_to_data[2000].is_asleep
    assert worker.sleep_mode_backend._state == "SUSPENDED"
    for ptr in (3000, 4000, 5000):
        assert allocator.pointer_to_data[ptr].contents == before[ptr]
    # A complete refit supplies fresh model weights; the old weight payload
    # was deliberately not backed up. Mixed allocations count in runtime bytes.
    allocator.pointer_to_data[1000].contents = b"n" * 100
    worker.restore_partial_kv(["kv_cache"])
    assert allocator.pointer_to_data[2000].contents == before[2000]
    assert allocator.pointer_to_data[1000].contents == b"n" * 100
    assert all(not row.is_asleep and row.cpu_backup_tensor is None for row in allocator.pointer_to_data.values())
    assert set(allocator.pointer_to_data) == set(before)


def test_discard_boundary_still_protects_sampler_state(worker_fixture):
    worker, allocator, _ = worker_fixture
    receipt = worker.offload_partial_kv(False, 0)
    assert receipt["pools"]["kv_cache"]["cpu_backup_bytes"] == 0
    assert receipt["pools"][memory.RUNTIME_TAG]["cpu_backup_bytes"] == 64
    worker.restore_partial_kv(["weights"])
    worker.restore_partial_kv(["kv_cache"])
    assert allocator.pointer_to_data[3000].contents == bytes([3]) * 64


def test_low_host_memory_never_authorizes_restore(worker_fixture, monkeypatch):
    worker, allocator, _ = worker_fixture
    original = worker.sleep_mode_backend.suspend
    monkeypatch.setattr(memory, "host_memory_available", lambda: 239)
    with pytest.raises(RuntimeError, match="Insufficient"):
        worker.offload_partial_kv(True, 0)
    assert allocator.copies == 0
    assert worker.sleep_mode_backend.suspend == original
    with pytest.raises(RuntimeError, match="acknowledged"):
        worker.restore_partial_kv(["weights"])


def test_short_cpu_backup_fails_receipt(worker_fixture):
    worker, allocator, _ = worker_fixture
    original = allocator.sleep

    def incomplete(offload_tags):
        original(offload_tags)
        allocator.pointer_to_data[2000].cpu_backup_tensor = _Backup(b"short")

    allocator.sleep = incomplete
    with pytest.raises(RuntimeError, match="incomplete"):
        worker.offload_partial_kv(True, 0)
    with pytest.raises(RuntimeError, match="acknowledged"):
        worker.restore_partial_kv(["kv_cache"])


def test_restore_detects_missing_runtime_state(worker_fixture):
    worker, allocator, _ = worker_fixture
    worker.offload_partial_kv(True, 0)
    allocator.wake_up = lambda tags: None
    with pytest.raises(RuntimeError, match="request state"):
        worker.restore_partial_kv(["weights"])


def test_view_offsets_endpoints_and_shared_ownership():
    entries = memory.allocations_for_views(
        [("outside", 99), ("a", 100), ("b", 115), ("end", 116), ("kv", 203)],
        [(200, 8, "kv_cache"), (100, 16, "weights")],
    )
    assert len(entries) == 1
    assert entries[0]["pointer"] == 100
    assert entries[0]["bytes"] == 16
    assert entries[0]["paths"] == ["a", "b"]


def test_protection_is_idempotent_and_validates_before_retagging(worker_fixture):
    worker, allocator, _ = worker_fixture
    first = memory.protect_runtime_state(worker.model_runner, allocator)
    second = memory.protect_runtime_state(worker.model_runner, allocator)
    assert first["protected_allocation_bytes"] == second["protected_allocation_bytes"] == 64
    allocator.pointer_to_data[3000].is_asleep = True
    with pytest.raises(RuntimeError, match="before"):
        memory.protect_runtime_state(worker.model_runner, allocator)


def test_unknown_runner_or_missing_sampler_is_rejected(worker_fixture):
    worker, _, _ = worker_fixture
    with pytest.raises(RuntimeError, match="Unsupported runner"):
        memory.runtime_tensors(SimpleNamespace())
    worker.model_runner.sampler = SimpleNamespace()
    with pytest.raises(RuntimeError, match="sampler"):
        memory.runtime_tensors(worker.model_runner)


def test_dense_v2_default_state_can_be_empty_but_sampler_is_protected(worker_fixture):
    worker, allocator, _ = worker_fixture
    default_state = type("DefaultModelState", (), {"__module__": "vllm.v1.worker.gpu.model_states.default"})
    worker.model_runner.model_state = default_state()
    receipt = memory.protect_runtime_state(worker.model_runner, allocator)
    assert receipt["protected_allocation_bytes"] == 64
    assert allocator.pointer_to_data[3000].tag == memory.RUNTIME_TAG
    worker.model_runner.sampler = SimpleNamespace()
    with pytest.raises(RuntimeError, match="sampler"):
        memory.runtime_tensors(worker.model_runner)


def test_cpu_admission_lock_is_released_on_success_and_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "host_memory_available", lambda: 200)
    with memory.admit_cpu_copy(100, 100, directory=tmp_path):
        with pytest.raises(TimeoutError):
            with memory.admit_cpu_copy(1, 0, directory=tmp_path, timeout=0):
                raise AssertionError("A second copy must not enter")
    with pytest.raises(RuntimeError, match="Insufficient"):
        with memory.admit_cpu_copy(101, 100, directory=tmp_path):
            raise AssertionError("Out-of-budget copy must not enter")
    with memory.admit_cpu_copy(100, 100, directory=tmp_path):
        pass


@pytest.mark.parametrize("limit,used,expected", [("1000", "700", 300), ("max", "700", 2048)])
def test_host_budget_respects_cgroup(monkeypatch, limit, used, expected):
    data = {
        "/proc/meminfo": "MemAvailable: 2 kB\n",
        "/sys/fs/cgroup/memory.max": limit,
        "/sys/fs/cgroup/memory.current": used,
    }
    monkeypatch.setattr(memory.Path, "is_file", lambda path: str(path) in data)
    monkeypatch.setattr(memory.Path, "read_text", lambda path: data[str(path)])
    assert memory.host_memory_available() == expected
