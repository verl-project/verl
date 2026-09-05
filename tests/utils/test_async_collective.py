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

import gc
import importlib.util
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_MODULE_PATH = Path(__file__).parents[2] / "verl" / "utils" / "collective.py"
_SPEC = importlib.util.spec_from_file_location("collective_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
collective = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = collective
_SPEC.loader.exec_module(collective)


class _FakeWork:
    def __init__(self, callback=lambda: None):
        self.callback = callback
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1
        self.callback()


class _FakeEvent:
    def __init__(self, order, name="complete_event"):
        self.order = order
        self.name = name

    def record(self):
        self.order.append(self.name)


class _FakeGroup:
    group_name = "sp-group-0"


def _run_gloo_collective(rank, world_size, init_method):
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    try:
        tensor = torch.tensor(float(rank + 1))
        handle = collective.AsyncCollectiveHandle(
            work=dist.all_reduce(tensor, async_op=True),
            finalize=tensor.clone,
            comm_kind="all_reduce",
            process_group_id="world",
            sequence_id=0,
        )
        result = handle.wait()
        assert result.item() == 3.0
        assert handle.collective_complete
        assert handle.finalized
    finally:
        dist.destroy_process_group()


def _load_ulysses_module(monkeypatch):
    verl_package = types.ModuleType("verl")
    verl_package.__path__ = []
    utils_package = types.ModuleType("verl.utils")
    utils_package.__path__ = []
    monkeypatch.setitem(sys.modules, "verl", verl_package)
    monkeypatch.setitem(sys.modules, "verl.utils", utils_package)
    monkeypatch.setitem(sys.modules, "verl.utils.collective", collective)

    module_path = _MODULE_PATH.with_name("ulysses.py")
    spec = importlib.util.spec_from_file_location("ulysses_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handle_separates_collective_completion_from_finalize():
    order = []
    work = _FakeWork(lambda: order.append("work"))
    result = object()

    def finalize():
        order.append("finalize")
        return result

    handle = collective.AsyncCollectiveHandle(
        work=work,
        finalize=finalize,
        comm_kind="all_to_all",
        process_group_id="sp-group-0",
        sequence_id=4,
        complete_event=_FakeEvent(order),
    )

    handle.wait_collective()
    handle.wait_collective()
    assert order == ["work", "complete_event"]
    assert handle.collective_complete
    assert not handle.finalized

    assert handle.finalize_result() is result
    assert handle.wait() is result
    assert order == ["work", "complete_event", "finalize"]
    assert work.wait_count == 1
    assert handle.finalized


def test_handle_rejects_invalid_completion_event():
    handle = collective.AsyncCollectiveHandle(
        work=_FakeWork(),
        finalize=lambda: None,
        comm_kind="all_reduce",
        process_group_id="dp-group-0",
        sequence_id=0,
        complete_event=object(),
    )
    with pytest.raises(TypeError, match=r"record\(\)"):
        handle.wait_collective()
    assert handle.work.wait_count == 0


def test_handle_caches_finalizer_error_without_repeating_side_effects():
    calls = 0
    error = RuntimeError("layout failed")

    def fail_finalize():
        nonlocal calls
        calls += 1
        raise error

    work = _FakeWork()
    handle = collective.AsyncCollectiveHandle(
        work=work,
        finalize=fail_finalize,
        comm_kind="all_gather",
        process_group_id="sp-group-0",
        sequence_id=0,
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="layout failed") as raised:
            handle.wait()
        assert raised.value is error
    assert work.wait_count == 1
    assert calls == 1
    assert handle.finalization_attempted
    assert not handle.finalized
    assert handle.finalization_error is error


def test_handle_rejects_a_second_cuda_consumer_stream(monkeypatch):
    current_stream = [(0, 11)]
    monkeypatch.setattr(collective, "_accelerator_stream_key", lambda _device: current_stream[0])
    finalize_calls = 0

    def finalize():
        nonlocal finalize_calls
        finalize_calls += 1
        return object()

    handle = collective.AsyncCollectiveHandle(
        work=_FakeWork(),
        finalize=finalize,
        comm_kind="all_reduce",
        process_group_id="dp-group-0",
        sequence_id=0,
        consumer_device=torch.device("cuda", 0),
    )

    handle.wait_collective()
    current_stream[0] = (0, 12)
    with pytest.raises(RuntimeError, match="one CUDA consumer stream"):
        handle.finalize_result()
    assert finalize_calls == 0

    current_stream[0] = (0, 11)
    result = handle.finalize_result()
    assert handle.wait() is result
    assert finalize_calls == 1


def test_handle_keeps_owned_resources_alive():
    class Resource:
        pass

    resource = Resource()
    reference = weakref.ref(resource)
    handle = collective.AsyncCollectiveHandle(
        work=_FakeWork(),
        finalize=lambda: None,
        comm_kind="all_reduce",
        process_group_id="dp-group-0",
        sequence_id=0,
        owned_resources=(resource,),
    )
    del resource
    gc.collect()
    assert reference() is not None
    del handle
    gc.collect()
    assert reference() is None


def test_handle_requires_immutable_owned_resource_container():
    with pytest.raises(TypeError, match="immutable tuple"):
        collective.AsyncCollectiveHandle(
            work=_FakeWork(),
            finalize=lambda: None,
            comm_kind="all_reduce",
            process_group_id="dp-group-0",
            sequence_id=0,
            owned_resources=[],
        )


def test_sequence_ids_are_monotonic_per_process_group():
    group_a = _FakeGroup()
    group_b = _FakeGroup()
    a0 = collective.next_collective_sequence_id(group_a)
    a1 = collective.next_collective_sequence_id(group_a)
    b0 = collective.next_collective_sequence_id(group_b)
    assert a1 == a0 + 1
    assert b0 == 0
    assert collective.resolve_process_group_id(group_a) == "sp-group-0"


def test_handle_wraps_real_gloo_work(tmp_path):
    init_method = f"file://{tmp_path / 'gloo_init'}"
    mp.spawn(_run_gloo_collective, args=(2, init_method), nprocs=2, join=True)


def test_ulysses_async_all_to_all_returns_structured_handle(monkeypatch):
    ulysses = _load_ulysses_module(monkeypatch)
    group = _FakeGroup()
    works = []
    order = []
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda group=None: 2)

    def fake_all_to_all(output_list, input_list, *, group, async_op):
        assert async_op
        order.append("collective_launch")

        def complete():
            order.append("work_complete")
            for output, input_ in zip(output_list, input_list, strict=True):
                output.copy_(input_)

        work = _FakeWork(complete)
        works.append(work)
        return work

    monkeypatch.setattr(ulysses.dist, "all_to_all", fake_all_to_all)
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    handle = ulysses.launch_all_to_all_tensor(
        tensor,
        scatter_dim=0,
        gather_dim=0,
        group=group,
        launch_event=_FakeEvent(order, "launch_event"),
        complete_event=_FakeEvent(order),
    )
    assert isinstance(handle, collective.AsyncCollectiveHandle)
    assert handle.comm_kind == "ulysses_all_to_all"
    assert handle.process_group_id == "sp-group-0"
    assert handle.consumer_device == tensor.device
    assert len(handle.owned_resources) == 5
    assert torch.equal(handle.wait(), tensor)
    assert works[0].wait_count == 1
    assert order == ["launch_event", "collective_launch", "work_complete", "complete_event"]

    legacy_wait = ulysses.all_to_all_tensor(tensor, scatter_dim=0, gather_dim=0, group=group, async_op=True)
    assert callable(legacy_wait)
    assert torch.equal(legacy_wait(), tensor)
    assert works[1].wait_count == 1


def test_ulysses_legacy_async_closure_preserves_wait_errors(monkeypatch):
    ulysses = _load_ulysses_module(monkeypatch)
    group = _FakeGroup()
    error = RuntimeError("collective failed")

    def fail_wait():
        raise error

    work = _FakeWork(fail_wait)
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        ulysses.dist,
        "all_to_all",
        lambda output_list, input_list, *, group, async_op: work,
    )

    legacy_wait = ulysses.all_to_all_tensor(
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        scatter_dim=0,
        gather_dim=0,
        group=group,
        async_op=True,
    )
    assert callable(legacy_wait)
    with pytest.raises(RuntimeError, match="collective failed") as raised:
        legacy_wait()
    assert raised.value is error
    assert work.wait_count == 1


def test_ulysses_async_all_gather_returns_structured_handle(monkeypatch):
    ulysses = _load_ulysses_module(monkeypatch)
    group = _FakeGroup()
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda group=None: 2)

    def fake_all_gather_into_tensor(output, input_, *, group, async_op):
        assert async_op

        def complete():
            output[: input_.shape[0]].copy_(input_)
            output[input_.shape[0] :].copy_(input_)

        return _FakeWork(complete)

    monkeypatch.setattr(ulysses.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    handle = ulysses.launch_all_gather_tensor(tensor, group)

    assert handle.comm_kind == "ulysses_all_gather"
    assert handle.process_group_id == "sp-group-0"
    assert handle.consumer_device == tensor.device
    assert len(handle.owned_resources) == 2
    assert handle.owned_resources[0] is tensor
    assert torch.equal(handle.wait(), torch.cat((tensor, tensor)))
