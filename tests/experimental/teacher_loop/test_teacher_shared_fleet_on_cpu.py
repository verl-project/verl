# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import asyncio
import os
import shutil
import tempfile

import pytest
import torch
from torch import nn

from verl.experimental.teacher_loop.slot_scheduler import SharedTeacherLeases
from verl.experimental.teacher_loop.teacher_weight_cache import (
    apply_shard_to_models,
    cleanup_cache_dir,
    estimate_model_bytes,
    get_tp_rank,
    load_shard_mmap,
    make_job_cache_dir,
    make_teacher_cache_dir,
    prepare_cache_dir,
    resolve_cache_root,
    tp_shard_dir,
    write_shard_from_models,
)

SHM_ROOT = "/dev/shm"


class _TinyModel(nn.Module):
    """Small module with two parameters and one buffer, enough to exercise the cache."""

    def __init__(self, *, seed: int = 0, in_features: int = 4):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.fc = nn.Linear(in_features, 3, bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(3, in_features, generator=generator))
            self.fc.bias.copy_(torch.randn(3, generator=generator))
        self.register_buffer("running", torch.arange(2, dtype=torch.float32))


class _SquareModel(nn.Module):
    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(rows, cols))


def _all_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    tensors = dict(model.named_parameters())
    tensors.update(model.named_buffers())
    return tensors


def _new_shard_dir(base: str, teacher_key: str = "teacher_a", tp_rank: int = 0) -> str:
    job_dir = make_job_cache_dir(base, "job1")
    return tp_shard_dir(make_teacher_cache_dir(job_dir, teacher_key), tp_rank)


# ---------------------------------------------------------------------------
# Directory layout / capacity
# ---------------------------------------------------------------------------
def test_cache_dir_layout_and_backends(tmp_path):
    assert resolve_cache_root("disk", str(tmp_path)) == os.path.abspath(str(tmp_path))
    assert resolve_cache_root("shm") == SHM_ROOT + "/verl_teacher_cache"
    # teacher_cache_dir is meaningless for shm and must not change the layout
    assert resolve_cache_root("shm", str(tmp_path)) == SHM_ROOT + "/verl_teacher_cache"

    with pytest.raises(ValueError, match="Unknown teacher_cache_backend"):
        resolve_cache_root("redis")

    job_dir = make_job_cache_dir(str(tmp_path), "job/with spaces")
    assert job_dir == os.path.join(os.path.abspath(str(tmp_path)), "job_with_spaces")
    teacher_dir = make_teacher_cache_dir(job_dir, "teacher/a")
    assert teacher_dir == os.path.join(job_dir, "teacher_a")
    assert tp_shard_dir(teacher_dir, 2) == os.path.join(teacher_dir, "tp2")


def test_get_tp_rank_maps_local_rank():
    class ParallelConfig:
        tensor_parallel_size = 4

    assert get_tp_rank(0, None) == 0
    assert get_tp_rank(4, None) == 0
    assert get_tp_rank(5, ParallelConfig()) == 1
    assert get_tp_rank(7, ParallelConfig()) == 3


def test_prepare_cache_dir_creates_dir_and_reports_space(tmp_path):
    cache_dir = os.path.join(str(tmp_path), "nested", "cache")
    info = prepare_cache_dir(cache_dir)
    assert os.path.isdir(cache_dir)
    assert info["path"] == os.path.abspath(cache_dir)
    assert info["free_bytes"] > 0
    assert info["total_bytes"] >= info["free_bytes"]


def test_prepare_cache_dir_fails_when_not_enough_space(tmp_path):
    usage = shutil.disk_usage(str(tmp_path))
    with pytest.raises(RuntimeError) as exc_info:
        prepare_cache_dir(str(tmp_path), min_free_bytes=usage.free + (1 << 40))
    message = str(exc_info.value)
    assert "GiB" in message
    assert "teacher_cache_dir" in message


def test_cleanup_refuses_unsafe_paths():
    for unsafe in ("/", "/tmp", "/dev/shm", SHM_ROOT):
        assert cleanup_cache_dir(unsafe) is False
    assert cleanup_cache_dir(None) is False
    assert cleanup_cache_dir("") is False


def test_cleanup_removes_job_dir_once(tmp_path):
    job_dir = make_job_cache_dir(str(tmp_path), "job1")
    os.makedirs(tp_shard_dir(make_teacher_cache_dir(job_dir, "teacher_a"), 0))
    assert cleanup_cache_dir(job_dir) is True
    assert not os.path.exists(job_dir)
    assert cleanup_cache_dir(job_dir) is False


# ---------------------------------------------------------------------------
# Snapshot / mmap roundtrip
# ---------------------------------------------------------------------------
def test_estimate_model_bytes_matches_tensor_size():
    model = _TinyModel()
    expected = sum(t.numel() * t.element_size() for t in _all_tensors(model).values())
    assert estimate_model_bytes([model]) == expected


def test_cache_roundtrip_restores_weights_and_buffers(tmp_path):
    model = _TinyModel(seed=1)
    before = {name: tensor.detach().clone() for name, tensor in _all_tensors(model).items()}

    shard_dir = _new_shard_dir(str(tmp_path))
    written = write_shard_from_models([model], shard_dir)

    assert written > 0
    assert os.path.getsize(os.path.join(shard_dir, "weights.bin")) == written
    assert os.path.isfile(os.path.join(shard_dir, "meta.json"))

    # The snapshot lives in the file, not in the live tensors.
    with torch.no_grad():
        for tensor in _all_tensors(model).values():
            tensor.add_(1.0)

    shard = load_shard_mmap(shard_dir)
    for name, tensor in before.items():
        cached = shard["params"].get(name, shard["buffers"].get(name))
        torch.testing.assert_close(cached, tensor)

    num_params, num_buffers = apply_shard_to_models([model], shard)
    assert (num_params, num_buffers) == (2, 1)
    for name, tensor in _all_tensors(model).items():
        torch.testing.assert_close(tensor, before[name])


def test_map_rejects_shard_built_for_another_checkpoint(tmp_path):
    model = _TinyModel(seed=1)
    shard_dir = _new_shard_dir(str(tmp_path))
    teacher_a = str(tmp_path / "teacher_a")
    write_shard_from_models([model], shard_dir, model_path=teacher_a)

    assert load_shard_mmap(shard_dir, model_path=teacher_a)["params"]
    with pytest.raises(RuntimeError, match="stale teacher weight cache"):
        load_shard_mmap(shard_dir, model_path=str(tmp_path / "teacher_b"))


def test_cache_roundtrip_multiple_models_uses_prefix(tmp_path):
    models = [_TinyModel(seed=1), _TinyModel(seed=2)]
    before = [{name: t.detach().clone() for name, t in _all_tensors(m).items()} for m in models]

    shard_dir = _new_shard_dir(str(tmp_path))
    write_shard_from_models(models, shard_dir)
    shard = load_shard_mmap(shard_dir)

    assert any(name.startswith("__model1__") for name in shard["params"])
    assert "__model1__fc.weight" in shard["params"]
    assert "__model1__running" in shard["buffers"]

    for model in models:
        with torch.no_grad():
            for tensor in _all_tensors(model).values():
                tensor.zero_()

    num_params, num_buffers = apply_shard_to_models(models, load_shard_mmap(shard_dir))
    assert (num_params, num_buffers) == (4, 2)
    for model, expected in zip(models, before, strict=True):
        for name, tensor in _all_tensors(model).items():
            torch.testing.assert_close(tensor, expected[name])


def test_apply_transposed_shard(tmp_path):
    source = _SquareModel(3, 5)
    shard_dir = _new_shard_dir(str(tmp_path))
    write_shard_from_models([source], shard_dir)

    target = _SquareModel(5, 3)
    apply_shard_to_models([target], load_shard_mmap(shard_dir))
    torch.testing.assert_close(target.w, source.w.detach().transpose(-2, -1))


def test_apply_rejects_incompatible_shape(tmp_path):
    source = _SquareModel(3, 5)
    shard_dir = _new_shard_dir(str(tmp_path))
    write_shard_from_models([source], shard_dir)

    target = _SquareModel(2, 2)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        apply_shard_to_models([target], load_shard_mmap(shard_dir))


@pytest.mark.skipif(
    not (os.path.isdir(SHM_ROOT) and os.access(SHM_ROOT, os.W_OK)),
    reason="/dev/shm is not writable on this platform",
)
def test_cache_roundtrip_on_shm():
    base = tempfile.mkdtemp(dir=SHM_ROOT)
    try:
        model = _TinyModel(seed=3)
        before = {name: tensor.detach().clone() for name, tensor in _all_tensors(model).items()}
        shard_dir = os.path.join(base, "tp0")
        write_shard_from_models([model], shard_dir)

        with torch.no_grad():
            for tensor in _all_tensors(model).values():
                tensor.add_(2.0)
        apply_shard_to_models([model], load_shard_mmap(shard_dir))
        for name, tensor in _all_tensors(model).items():
            torch.testing.assert_close(tensor, before[name])
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------------------
# Occupancy leases (pure asyncio, no ray/vLLM)
# ---------------------------------------------------------------------------
class _Recorder:
    def __init__(self):
        self.calls: list[tuple[int, str]] = []

    async def __call__(self, slot: int, teacher_key: str) -> None:
        self.calls.append((slot, teacher_key))


@pytest.mark.asyncio
async def test_leases_first_wave_hit_does_not_apply():
    recorder = _Recorder()
    leases = SharedTeacherLeases(recorder, num_slots=1, key_priority=["teacher_a"])
    await leases.set_step_key_counts({"teacher_a": 1})

    slot = await leases.acquire("teacher_a")
    assert slot == 0
    assert leases.inflight == [1]
    assert recorder.calls == []
    assert await leases.release(slot) is True


@pytest.mark.asyncio
async def test_occupancy_keeps_first_wave_until_step_done():
    """4 slots / 8 keys: first wave stays until those teachers finish the step."""
    recorder = _Recorder()
    keys = [f"t{i}" for i in range(8)]
    leases = SharedTeacherLeases(recorder, num_slots=4, key_priority=keys)
    await leases.set_step_key_counts({key: 1 for key in keys})
    assert leases.keys == keys[:4]

    first = [await leases.acquire(key) for key in keys[:4]]
    assert first == [0, 1, 2, 3]
    assert recorder.calls == []

    pending = asyncio.create_task(leases.acquire("t4"))
    for _ in range(10):
        await asyncio.sleep(0)
    assert not pending.done()

    await leases.release(0)
    assert await asyncio.wait_for(pending, timeout=5) == 0
    assert recorder.calls == [(0, "t4")]
    assert leases.keys[0] == "t4"
    await leases.release(0)


@pytest.mark.asyncio
async def test_occupancy_donates_unstarted_in_script_order():
    recorder = _Recorder()
    keys = ["a", "b", "c", "d"]
    leases = SharedTeacherLeases(recorder, num_slots=2, key_priority=keys)
    await leases.set_step_key_counts({key: 1 for key in keys})
    assert leases.keys == ["a", "b"]

    slot_a = await leases.acquire("a")
    slot_b = await leases.acquire("b")
    waiter_c = asyncio.create_task(leases.acquire("c"))
    waiter_d = asyncio.create_task(leases.acquire("d"))
    for _ in range(10):
        await asyncio.sleep(0)
    assert not waiter_c.done() and not waiter_d.done()

    await leases.release(slot_a)
    got_c = await asyncio.wait_for(waiter_c, timeout=5)
    assert got_c == slot_a
    assert recorder.calls[0] == (slot_a, "c")

    await leases.release(slot_b)
    got_d = await asyncio.wait_for(waiter_d, timeout=5)
    assert got_d == slot_b
    assert (slot_b, "d") in recorder.calls
    await leases.release(got_c)
    await leases.release(got_d)


@pytest.mark.asyncio
async def test_occupancy_second_step_resets_outstanding_and_donates():
    """A new generate dispatch must not inherit leftover remaining.

    Additive leftover on a first-wave key keeps ``should_reassign`` false, so
    later keys wait forever. That is the 1104_5 step-2 hang.
    """
    recorder = _Recorder()
    keys = [f"t{i}" for i in range(8)]
    leases = SharedTeacherLeases(recorder, num_slots=4, key_priority=keys)

    async def run_step(counts: dict, actual: dict | None = None) -> None:
        await leases.set_step_key_counts(counts)
        actual = actual or counts

        async def worker(key: str, n: int) -> None:
            for _ in range(n):
                slot = await leases.acquire(key)
                await leases.release(slot)

        await asyncio.wait_for(
            asyncio.gather(*(worker(key, actual[key]) for key in keys)),
            timeout=5,
        )

    await run_step({key: 3 for key in keys})
    # Previous step leftover must not block this step's donate.
    leases._scheduler.step_outstanding["t0"] = leases._scheduler.step_outstanding.get("t0", 0) + 2
    await run_step({key: 3 for key in keys})
    assert leases.inflight == [0, 0, 0, 0]
    assert leases._scheduler.step_outstanding == {}


@pytest.mark.asyncio
async def test_occupancy_leftover_goes_to_longest_queue():
    recorder = _Recorder()
    leases = SharedTeacherLeases(recorder, num_slots=2, key_priority=["short", "long"])
    await leases.set_step_key_counts({"short": 1, "long": 3})

    short_slot = await leases.acquire("short")
    long_slot = await leases.acquire("long")
    extra = asyncio.create_task(leases.acquire("long"))
    for _ in range(10):
        await asyncio.sleep(0)
    # long may stack on its holder (use) or wait for donate; either way short
    # finishing the step donates the leftover replica to long.
    await leases.release(short_slot)
    second_long = await asyncio.wait_for(extra, timeout=5)
    if second_long != long_slot:
        assert recorder.calls == [(short_slot, "long")]
        await leases.release(second_long)
    await leases.release(long_slot)


@pytest.mark.asyncio
async def test_leases_does_not_swap_a_busy_slot():
    recorder = _Recorder()
    leases = SharedTeacherLeases(recorder, num_slots=1, key_priority=["teacher_a", "teacher_b"])
    await leases.set_step_key_counts({"teacher_a": 1, "teacher_b": 1})

    slot = await leases.acquire("teacher_a")
    pending = asyncio.create_task(leases.acquire("teacher_b"))
    for _ in range(5):
        await asyncio.sleep(0)

    assert not pending.done()
    assert recorder.calls == []

    await leases.release(slot)
    assert await asyncio.wait_for(pending, timeout=5) == 0
    assert recorder.calls == [(0, "teacher_b")]
    await leases.release(0)


@pytest.mark.asyncio
async def test_leases_skip_failed_slot():
    async def apply_fn(slot: int, teacher_key: str) -> None:
        if slot == 0:
            raise RuntimeError("boom")

    leases = SharedTeacherLeases(apply_fn, num_slots=2, key_priority=["teacher_a"])
    with pytest.raises(RuntimeError, match="boom"):
        await leases.set_step_key_counts({"teacher_b": 1})
    assert 0 in leases.failed

    assert await leases.acquire("teacher_b") == 1
    await leases.release(1)


@pytest.mark.asyncio
async def test_leases_raises_when_every_slot_failed():
    async def apply_fn(slot: int, teacher_key: str) -> None:
        raise RuntimeError("canvas is broken")

    leases = SharedTeacherLeases(apply_fn, num_slots=1, key_priority=["teacher_a"])
    with pytest.raises(RuntimeError, match="canvas is broken"):
        await leases.set_step_key_counts({"teacher_b": 1})
    with pytest.raises(RuntimeError, match="No usable teacher replica"):
        await leases.acquire("teacher_c")


@pytest.mark.asyncio
async def test_leases_never_swaps_a_leased_slot():
    num_slots = 2
    keys = ["teacher_a", "teacher_b", "teacher_c"]
    outstanding = [0] * num_slots
    swapped_while_busy: list[int] = []

    async def apply_fn(slot: int, teacher_key: str) -> None:
        if outstanding[slot] != 0:
            swapped_while_busy.append(slot)
        await asyncio.sleep(0)
        assert teacher_key in keys

    leases = SharedTeacherLeases(apply_fn, num_slots=num_slots, key_priority=keys)
    n_workers = 200
    counts = {key: 0 for key in keys}
    for i in range(n_workers):
        counts[keys[i % len(keys)]] += 1
    await leases.set_step_key_counts(counts)

    async def worker(index: int) -> None:
        teacher_key = keys[index % len(keys)]
        slot = await leases.acquire(teacher_key)
        outstanding[slot] += 1
        try:
            await asyncio.sleep(0)
            assert leases.keys[slot] == teacher_key
            await asyncio.sleep(0)
        finally:
            outstanding[slot] -= 1
            await leases.release(slot)

    await asyncio.wait_for(asyncio.gather(*(worker(i) for i in range(n_workers))), timeout=30)

    assert swapped_while_busy == []
    assert leases.inflight == [0, 0]


@pytest.mark.asyncio
async def test_leases_release_unknown_slot_is_ignored():
    recorder = _Recorder()
    leases = SharedTeacherLeases(recorder, num_slots=1, key_priority=["teacher_a"])
    assert await leases.release(5) is False
    assert await leases.release(-1) is False
    assert await leases.release(0) is False
    assert leases.inflight == [0]
