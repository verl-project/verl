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
import ast
import asyncio
import importlib
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import ray
import torch


@pytest.fixture
def backend(monkeypatch):
    from verl.checkpoint_engine.base import CheckpointEngineRegistry

    monkeypatch.setattr(CheckpointEngineRegistry, "_registry", dict(CheckpointEngineRegistry._registry))
    current = threading.local()
    current.device = 0
    calls = []

    @contextmanager
    def device(index):
        previous = getattr(current, "device", 0)
        current.device = index
        try:
            yield
        finally:
            current.device = previous

    class Array:
        def __init__(self, size=1):
            self.device = getattr(current, "device", 0)
            self.size = size

    class Group:
        def __init__(self, world_size, rank, group_name):
            self.world_size, self.rank, self.group_name = world_size, rank, group_name
            calls.append(("create", rank, group_name))

        def _generate_group_key(self, key):
            return f"{key}@{self.group_name}"

        def barrier(self):
            calls.append(("global-barrier",))

        def allreduce(self, tensors):
            calls.append(("barrier", [t.device for t in tensors], self._generate_group_key(str(tensors[0].device))))

        def broadcast(self, tensors, options):
            calls.append(("broadcast", getattr(current, "device", 0), tensors[0].device, options.root_rank))
            self._generate_group_key(str(tensors[0].device))

        def destroy_group(self):
            calls.append(("destroy", self.rank, self.group_name))

    path = Path(ray.__file__).parent / "util/collective/collective_group/nccl_collective_group.py"
    tree = ast.parse(path.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NCCLGroup")
    methods = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in ("barrier", "_generate_group_key")
    ]
    namespace = {
        "BarrierOptions": lambda: None,
        "nccl_util": SimpleNamespace(get_num_gpus=lambda: 8, Device=device),
        "cupy": SimpleNamespace(array=lambda value: Array()),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    Group.barrier = namespace["barrier"]
    Group._generate_group_key = namespace["_generate_group_key"]
    Group._used_gpu_indices = set()

    cupy = ModuleType("cupy")
    cupy.ndarray = Array
    cupy.cuda = SimpleNamespace(Device=device)
    cupy.float32, cupy.uint8 = "float32", "uint8"
    cupy.ones = cupy.zeros = lambda size, dtype: Array(size)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(
        sys.modules, "ray.util.collective.collective_group.nccl_collective_group", SimpleNamespace(NCCLGroup=Group)
    )
    monkeypatch.setitem(sys.modules, "ray.util.collective.types", SimpleNamespace(BroadcastOptions=SimpleNamespace))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: getattr(current, "device", 0))
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch, "zeros", lambda size, dtype, device: Array(size))
    name = "verl.checkpoint_engine.nccl_checkpoint_engine"
    previous = sys.modules.pop(name, None)
    module = importlib.import_module(name)
    monkeypatch.setattr(module.NCCLCheckpointEngine, "get_node_id", lambda self: "node")
    try:
        yield SimpleNamespace(module=module, current=current, calls=calls, Array=Array, original=Group)
    finally:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous
        package = sys.modules.get("verl.checkpoint_engine")
        if getattr(package, "nccl_checkpoint_engine", None) is module:
            delattr(package, "nccl_checkpoint_engine")


def engine(backend, **kwargs):
    return backend.module.NCCLCheckpointEngine(bucket_size=64, **kwargs)


@pytest.mark.parametrize("ordinals", [(0, 0), (0, 3)])
def test_rendezvous_is_independent_of_local_gpu_ordinals(backend, ordinals):
    groups = [backend.module._CheckpointNCCLGroup(2, rank, "ce", ordinal) for rank, ordinal in enumerate(ordinals)]
    assert [group._generate_group_key(str(device)) for group, device in zip(groups, ordinals, strict=True)] == [
        "single-device@ce"
    ] * 2
    with pytest.raises(ValueError, match="Checkpoint group expects GPU"):
        groups[0]._generate_group_key("0,1")
    assert backend.original._generate_group_key(groups[0], "3") == "3@ce"
    assert backend.original.barrier.__name__ == "barrier"


@pytest.mark.parametrize("ordinal", [0, 3])
def test_prepare_and_first_barrier_use_only_the_engine_device(backend, ordinal):
    backend.current.device = ordinal
    value = engine(backend)
    value.prepare()
    backend.current.device = 0
    value.init_process_group(0, 2, None, 1)
    assert value.device == ordinal
    assert value.send_buf.device == ordinal
    assert ("barrier", [ordinal], "single-device@default") in backend.calls
    assert ("global-barrier",) not in backend.calls


def test_rpc_executor_thread_rebinds_the_engine_device(backend):
    backend.current.device = 3
    value = engine(backend)
    value.prepare()
    bucket = value.send_buf
    value.init_process_group(0, 2, None, 1)
    socket = SimpleNamespace(send_string=lambda *args, **kwargs: None, send_pyobj=lambda value: None)

    async def run():
        operation = backend.module.BroadcastOperation(0, value.group_name, bucket, {}, socket, "meta", value._broadcast)
        await operation.wait_for_complete()

    asyncio.run(run())
    assert ("broadcast", 3, 3, 0) in backend.calls


def test_broadcast_rejects_a_tensor_on_another_gpu(backend):
    value = engine(backend, device=3)
    value.init_process_group(0, 2, None, 1)
    with pytest.raises(ValueError, match="expects GPU 3"):
        value._broadcast(backend.Array())


def test_reuse_preserves_group_and_rejects_changed_topology(backend):
    value = engine(backend)
    value.init_process_group(0, 2, None, 1)
    group = value._group
    value.init_process_group(0, 2, None, 1)
    assert value._group is group
    with pytest.raises(AssertionError, match="world_size"):
        value.init_process_group(0, 3, None, 1)
    with pytest.raises(RuntimeError, match="owned by another engine"):
        engine(backend).init_process_group(0, 2, None, 1)


def test_rebuild_destroys_only_the_owned_group(backend):
    value = engine(backend, rebuild_group=True)
    value.init_process_group(0, 2, None, 1)
    other = engine(backend, group_name="other")
    other.init_process_group(0, 2, None, 1)
    value.finalize()
    assert backend.module._CHECKPOINT_GROUPS == {"other": other._group}
    value.init_process_group(0, 3, None, 1)
    assert value.world_size == 3
    assert [call for call in backend.calls if call[0] == "destroy"] == [("destroy", 0, "default")]


def test_excluded_rank_does_not_create_a_group(backend):
    value = engine(backend, rebuild_group=True)
    value.init_process_group(-1, 2, None, 1)
    value.finalize()
    assert backend.calls == []


def test_other_ray_groups_keep_their_original_barrier(backend):
    group = backend.original(2, 0, "unrelated")
    group.barrier()
    assert backend.calls[-1][1] == list(range(8))
    value = engine(backend)
    value.init_process_group(0, 2, None, 1)
    group.barrier()
    assert backend.calls[-1][1] == list(range(8))
