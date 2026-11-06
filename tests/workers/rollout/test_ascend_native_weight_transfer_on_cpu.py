# Copyright 2026 Individual Contributor: yeyunu
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

"""Protocol checks for bounded native Ascend IPC weight updates."""

import asyncio
import contextlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _unexpected_dependency_call(*args, **kwargs):
    raise AssertionError("The protocol test must replace this device/runtime operation")


def _dependency_stubs():
    """Provide import-only substitutes; tensor operations and ZeroMQ stay real.

    Never import verl's rollout package initializer: it requires an installed
    vLLM distribution, even though these protocol tests do not run an engine.
    """
    names = [
        "verl",
        "verl.utils",
        "verl.utils.device",
        "verl.workers",
        "verl.workers.rollout",
        "verl.workers.rollout.utils",
        "verl.workers.rollout.vllm_rollout",
        "ray",
        "vllm",
        "vllm.utils",
        "vllm.utils.network_utils",
        "vllm.distributed",
        "vllm.distributed.weight_transfer",
        "vllm.distributed.weight_transfer.base",
        "torch_npu",
        "torch_npu.multiprocessing",
        "torch_npu.multiprocessing.reductions",
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.weight_transfer",
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        "vllm_ascend.distributed.weight_transfer.packed_tensor",
    ]
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, module)
    device = modules["verl.utils.device"]
    device.get_device_id = lambda: 0
    device.get_device_name = lambda: "cpu"
    device.get_torch_device = lambda: SimpleNamespace(synchronize=lambda: None)
    device.is_support_ipc = False

    async def ensure_async_iterator(weights):
        if hasattr(weights, "__aiter__"):
            async for item in weights:
                yield item
        else:
            for item in weights:
                yield item

    modules["verl.workers.rollout.utils"].ensure_async_iterator = ensure_async_iterator
    modules["ray"].get_actor = _unexpected_dependency_call
    network = modules["vllm.utils.network_utils"]
    network.get_ip = _unexpected_dependency_call
    network.get_open_port = _unexpected_dependency_call
    modules["torch_npu.multiprocessing.reductions"].rebuild_npu_tensor = _unexpected_dependency_call
    modules["vllm_ascend.distributed.weight_transfer.npu_ipc_engine"].npu_generate_uuid = _unexpected_dependency_call

    def packed_consumer(ipc_handle, physical_npu_id, names, shapes, dtype_names, tensor_sizes, device_index):
        # Model the public ownership contract; a separate pinned-source audit
        # executes the real upstream helper and real layerwise deferred loader.
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        args = list(ipc_handle[physical_npu_id])
        args[6] = device_index
        data = rebuild_npu_tensor(*args)
        result, offset = [], 0
        for name, shape, dtype_name, size in zip(names, shapes, dtype_names, tensor_sizes, strict=True):
            result.append((name, data[offset : offset + size].clone().view(getattr(torch, dtype_name)).reshape(shape)))
            offset += size
        return result

    modules["vllm_ascend.distributed.weight_transfer.packed_tensor"].packed_npu_ipc_consumer = packed_consumer

    @dataclass(frozen=True)
    class ParamMeta:
        name: str
        dtype: torch.dtype
        shape: tuple

    modules["vllm.distributed.weight_transfer.base"].ParamMeta = ParamMeta

    class HCCLTrainerWeightTransferEngine:
        trainer_init = staticmethod(_unexpected_dependency_call)
        send_weights = _unexpected_dependency_call
        shutdown = _unexpected_dependency_call

    hccl = modules["vllm_ascend.distributed.weight_transfer.hccl_engine"]
    hccl.HCCLTrainerWeightTransferEngine = HCCLTrainerWeightTransferEngine
    hccl.HCCLTrainerInitInfo = SimpleNamespace
    return modules


@contextlib.contextmanager
def _isolated_modules(modules):
    """Restore every replaced sys.modules entry, including on import failures."""
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in modules}
    try:
        sys.modules.update(modules)
        yield
    finally:
        for name, previous in saved.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _load_native_protocol():
    modules = _dependency_stubs()
    for filename in ["bucketed_weight_transfer", "ascend_weight_transfer"]:
        name = f"verl.workers.rollout.vllm_rollout.{filename}"
        path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout" / f"{filename}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        modules[name] = importlib.util.module_from_spec(spec)
    with _isolated_modules(modules):
        for filename in ["bucketed_weight_transfer", "ascend_weight_transfer"]:
            name = f"verl.workers.rollout.vllm_rollout.{filename}"
            modules[name].__spec__.loader.exec_module(modules[name])
    return modules["verl.workers.rollout.vllm_rollout.ascend_weight_transfer"]


native = _load_native_protocol()


@pytest.fixture(autouse=True)
def _isolated_runtime_imports():
    # Runtime imports inside reload/HCCL functions need the same scoped isolation.
    with _isolated_modules(_dependency_stubs()):
        yield


class ReplySocket:
    def __init__(self, reply=None):
        self.reply = {"ok": True} if reply is None else reply
        self.messages = []

    def send_pyobj(self, message):
        self.messages.append(message)

    def recv_pyobj(self):
        return self.reply


def _cpu_sender(monkeypatch, bucket_bytes=16):
    sender = native.AscendNativeIPCWeightSender("unused", bucket_size_mb=bucket_bytes / (1024 * 1024))
    # Production config uses whole MiB; shrink the buffer explicitly for tests.
    sender.bucket_size = bucket_bytes
    sender.buffer = torch.zeros(bucket_bytes, dtype=torch.uint8)
    sender.npu_uuid = "uuid"
    sender.receiver_started = True
    registry = {}

    def reduce(tensor):
        key = len(registry)
        registry[key] = tensor
        return None, (key, 0, 0, 0, 0, 0, 0)

    monkeypatch.setattr(native, "reduce_tensor", reduce)
    monkeypatch.setattr(sender, "_init_socket", lambda: None)
    monkeypatch.setattr(sender, "_init_buffer", lambda: None)
    monkeypatch.setattr(sender, "close", lambda: None)
    monkeypatch.setattr(native, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: None))
    return sender, registry


def test_packed_sender_exports_one_reference_for_each_nonempty_message(monkeypatch):
    sender, registry = _cpu_sender(monkeypatch, bucket_bytes=8)
    consumed = set()

    class Socket(ReplySocket):
        def send_pyobj(self, message):
            info = message["update_info"]
            if info["names"]:
                reference = info["ipc_handles"]["uuid"][0]
                assert reference not in consumed, "An IPC export reference must be consumed only once"
                consumed.add(reference)
            super().send_pyobj(message)

    sender.socket = Socket()
    asyncio.run(sender.async_send_weights([(str(i), torch.ones(2)) for i in range(3)]))
    assert len(consumed) == len(registry) == 3


def test_empty_terminal_message_does_not_export_an_unconsumed_reference(monkeypatch):
    sender, registry = _cpu_sender(monkeypatch, bucket_bytes=8)
    sender.socket = ReplySocket()
    # The oversized tensor is sent directly, leaving an empty terminal bucket.
    asyncio.run(sender.async_send_weights([("large", torch.ones(3))]))
    terminal = sender.socket.messages[-1]
    assert terminal["is_last"] and terminal["update_info"]["names"] == []
    assert terminal["update_info"]["ipc_handles"] == {}
    assert len(registry) == 1


def test_async_packed_sender_preserves_mixed_dtypes_scalar_empty_and_large_noncontiguous(monkeypatch):
    import torch_npu.multiprocessing.reductions as reductions
    from vllm_ascend.distributed.weight_transfer.packed_tensor import packed_npu_ipc_consumer

    sender, registry = _cpu_sender(monkeypatch)
    monkeypatch.setattr(reductions, "rebuild_npu_tensor", lambda key, *args: registry[key])
    retained, messages = [], []

    class Socket(ReplySocket):
        def send_pyobj(self, message):
            messages.append(message)
            info = message["update_info"]
            assert isinstance(info["ipc_handles"], dict)
            assert info["tensor_sizes"] == [
                torch.empty(shape, dtype=getattr(torch, dtype)).nbytes
                for shape, dtype in zip(info["shapes"], info["dtype_names"], strict=True)
            ]
            retained.extend(
                packed_npu_ipc_consumer(
                    ipc_handle=info["ipc_handles"],
                    physical_npu_id="uuid",
                    names=info["names"],
                    shapes=info["shapes"],
                    dtype_names=info["dtype_names"],
                    tensor_sizes=info["tensor_sizes"],
                    device_index=0,
                )
            )

    sender.socket = Socket()
    expected = [
        ("bytes", torch.tensor([1, 2, 3], dtype=torch.int8)),
        ("scalar", torch.tensor(4.25)),
        ("zero", torch.empty((0, 3), dtype=torch.bfloat16)),
        ("bf16", torch.tensor([5, 6, 7], dtype=torch.bfloat16)),
        ("large", torch.arange(24, dtype=torch.float32).reshape(4, 6).t()),
        ("tail", torch.tensor([8, 9], dtype=torch.int64)),
    ]
    assert not expected[4][1].is_contiguous()
    asyncio.run(sender.async_send_weights(iter(expected)))
    # Deferred model loading: consume only after all sender buffers have been reused.
    sender.buffer.fill_(255)
    assert [name for name, _ in retained] == [name for name, _ in expected]
    for (_, actual), (_, wanted) in zip(retained, expected, strict=True):
        assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert messages[-1]["is_last"] is True
    assert sum(message["is_last"] for message in messages) == 1
    assert messages[0]["update_info"]["names"] == ["bytes", "scalar", "zero", "bf16"]
    assert messages[0]["update_info"]["tensor_sizes"] == [3, 4, 0, 6]
    assert any(sum(message["update_info"]["tensor_sizes"]) > 16 for message in messages)


@pytest.mark.parametrize("zero_only", [False, True])
def test_async_packed_sender_empty_and_zero_only_complete(monkeypatch, zero_only):
    sender, _ = _cpu_sender(monkeypatch)
    sender.socket = ReplySocket()
    source = [("empty", torch.empty(0))] if zero_only else []
    asyncio.run(sender.async_send_weights(iter(source)))
    assert len(sender.socket.messages) == 1
    info = sender.socket.messages[0]["update_info"]
    assert info["names"] == (["empty"] if zero_only else [])
    assert info["tensor_sizes"] == ([0] if zero_only else [])
    assert sender.socket.messages[0]["is_last"] is True


def test_async_packed_source_reuse_does_not_change_previous_batches(monkeypatch):
    import torch_npu.multiprocessing.reductions as reductions
    from vllm_ascend.distributed.weight_transfer.packed_tensor import packed_npu_ipc_consumer

    sender, registry = _cpu_sender(monkeypatch, bucket_bytes=8)
    monkeypatch.setattr(reductions, "rebuild_npu_tensor", lambda key, *args: registry[key])
    retained = []

    class Socket(ReplySocket):
        def send_pyobj(self, message):
            info = message["update_info"]
            retained.extend(
                packed_npu_ipc_consumer(
                    ipc_handle=info["ipc_handles"],
                    physical_npu_id="uuid",
                    names=info["names"],
                    shapes=info["shapes"],
                    dtype_names=info["dtype_names"],
                    tensor_sizes=info["tensor_sizes"],
                    device_index=0,
                )
            )

    sender.socket = Socket()

    async def source():
        storage = torch.zeros(2)
        for value in range(4):
            storage.fill_(value)
            yield str(value), storage

    asyncio.run(sender.async_send_weights(source()))
    assert [tensor.tolist() for _, tensor in retained] == [[float(value)] * 2 for value in range(4)]


def test_async_packed_source_failure_aborts_without_finalizing(monkeypatch):
    sender, _ = _cpu_sender(monkeypatch, bucket_bytes=8)

    class Socket(ReplySocket):
        def setsockopt(self, *args):
            pass

    sender.socket = Socket()

    async def source():
        yield "first", torch.ones(2)
        yield "second", torch.ones(2)
        raise ValueError("source failed after a sent bucket")

    with pytest.raises(ValueError, match="source failed"):
        asyncio.run(sender.async_send_weights(source()))
    assert sender.socket.messages[0]["update_info"]["names"] == ["first"]
    assert sender.socket.messages[0]["is_last"] is False
    assert sender.socket.messages[-1]["abort"] is True
    assert all(not item.get("is_last", False) for item in sender.socket.messages)


def test_duplicate_names_are_transmitted_in_separate_batches(monkeypatch):
    sender, _ = _cpu_sender(monkeypatch, bucket_bytes=32)
    sender.socket = ReplySocket()
    asyncio.run(sender.async_send_weights(iter([("same", torch.ones(1)), ("same", torch.zeros(1))])))
    assert [item["update_info"]["names"] for item in sender.socket.messages] == [["same"], ["same"]]


def test_sender_surfaces_native_loader_failure(monkeypatch):
    sender, _ = _cpu_sender(monkeypatch)
    sender.socket = ReplySocket({"ok": False, "error": "incomplete Step5 weights"})
    monkeypatch.setattr(native, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: None))
    with pytest.raises(RuntimeError, match="incomplete Step5 weights"):
        sender._send_bucket({}, is_last=True)


@pytest.mark.parametrize("fail_update", [False, True])
def test_native_lifecycle_and_acknowledgement_order(monkeypatch, fail_update):
    events = []
    messages = iter(
        [
            {"update_info": {"names": ["weight"]}, "is_last": False},
            {"update_info": {"names": []}, "is_last": True},
        ]
    )

    class Socket:
        def setsockopt(self, *args):
            pass

        def connect(self, *args):
            pass

        def recv_pyobj(self):
            return next(messages)

        def send_pyobj(self, value):
            events.append(("ack", value["ok"]))

        def close(self, **kwargs):
            events.append(("close",))

    class Worker:
        def init_weight_transfer_engine(self, info):
            events.append(("init", info))

        def start_weight_update(self):
            events.append(("start",))

        def update_weights(self, info):
            events.append(("update", info["names"]))
            if fail_update:
                raise ValueError("bad weight")

        def finish_weight_update(self):
            events.append(("finish",))

    socket = Socket()
    monkeypatch.setattr(native.zmq.Context, "instance", lambda: SimpleNamespace(socket=lambda kind: socket))
    monkeypatch.setattr(
        native, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: events.append(("sync",)))
    )
    if fail_update:
        with pytest.raises(ValueError, match="bad weight"):
            native.receive_ascend_native_weights(Worker(), "ipc:///tmp/unused-native-test")
        assert events == [("init", {"packed": True}), ("start",), ("update", ["weight"]), ("ack", False), ("close",)]
    else:
        native.receive_ascend_native_weights(Worker(), "ipc:///tmp/unused-native-test")
        assert events == [
            ("init", {"packed": True}),
            ("start",),
            ("update", ["weight"]),
            ("sync",),
            ("ack", True),
            ("finish",),
            ("sync",),
            ("ack", True),
            ("close",),
        ]


@pytest.mark.parametrize("fail_finish", [False, True])
@pytest.mark.parametrize("empty_terminal", [False, True])
def test_reload_consumes_whole_stream_and_acknowledges_after_finish(monkeypatch, fail_finish, empty_terminal):
    import torch_npu.multiprocessing.reductions as reductions
    from vllm_ascend.distributed.weight_transfer import npu_ipc_engine

    events = []
    storage = torch.tensor([1.0, 2.0])
    messages = iter(
        [
            {
                "update_info": {
                    "names": ["a"],
                    "shapes": [[2]],
                    "dtype_names": ["float32"],
                    "ipc_handles": {"uuid": (0, 0, 0, 0, 0, 0, 0)},
                    "tensor_sizes": [8],
                },
                "is_last": False,
            },
            {
                "update_info": {
                    "names": [] if empty_terminal else ["b"],
                    "shapes": [] if empty_terminal else [[2]],
                    "dtype_names": [] if empty_terminal else ["float32"],
                    "ipc_handles": {} if empty_terminal else {"uuid": (0, 0, 0, 0, 0, 0, 0)},
                    "tensor_sizes": [] if empty_terminal else [8],
                },
                "is_last": True,
            },
        ]
    )

    class Socket:
        def setsockopt(self, *args):
            pass

        def connect(self, *args):
            pass

        def recv_pyobj(self):
            return next(messages)

        def send_pyobj(self, value):
            events.append(("ack", value["ok"]))
            storage.fill_(9)

        def close(self, **kwargs):
            pass

    class Worker:
        device = torch.device("cpu:0")

        def reload_weights(self, weights_iterator, is_checkpoint_format):
            assert is_checkpoint_format
            events.append("reload")
            weights = dict(weights_iterator)
            assert weights["a"].tolist() == [1, 2]
            if empty_terminal:
                assert list(weights) == ["a"]
            else:
                assert weights["b"].tolist() == [9, 9]
            events.append("finish")
            if fail_finish:
                raise ValueError("finalize failed")

    monkeypatch.setattr(native.zmq.Context, "instance", lambda: SimpleNamespace(socket=lambda kind: Socket()))
    monkeypatch.setattr(native, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(npu_ipc_engine, "npu_generate_uuid", lambda *args: "uuid")
    monkeypatch.setattr(reductions, "rebuild_npu_tensor", lambda *args: storage.view(torch.uint8))
    if fail_finish:
        with pytest.raises(ValueError, match="finalize failed"):
            native.receive_ascend_reload_weights(Worker(), "unused")
        assert events == ["reload", ("ack", True), "finish", ("ack", False)]
    else:
        native.receive_ascend_reload_weights(Worker(), "unused")
        assert events == ["reload", ("ack", True), "finish", ("ack", True)]


@pytest.mark.parametrize("failure", [None, "source", "send", "finish", "empty"])
def test_hccl_native_calls_bounded_stream_and_peer_replica(monkeypatch, failure):
    import asyncio

    import ray
    import torch.distributed as dist
    import vllm.utils.network_utils as network
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerWeightTransferEngine as Engine,
    )

    events, received = [], []
    import threading

    closed = threading.Event()
    main_thread = threading.get_ident()

    class Future:
        def __await__(self):
            async def done():
                return None

            return done().__await__()

    class RPC:
        def remote(self, method, **kwargs):
            assert threading.get_ident() == main_thread
            events.append(method)
            if failure == "finish" and method == "finish_weight_update":
                raise ValueError("finish failed")
            return Future()

    server = SimpleNamespace(collective_rpc=RPC())

    def actor(name):
        assert name == "vllm_server_1_0"
        return server

    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "4")
    monkeypatch.setattr(ray, "get_actor", actor)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(network, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(network, "get_open_port", lambda: 54321)
    monkeypatch.setattr(
        native,
        "get_torch_device",
        lambda: SimpleNamespace(synchronize=lambda: None, device=lambda device: contextlib.nullcontext()),
    )

    class Trainer:
        device = torch.device("cpu")

        def __init__(self, client, source):
            self.client, self.source = client, source

        def send_weights(self):
            assert threading.get_ident() != main_thread
            pairs = list(self.source)
            metadata = self.source.metadata()
            assert [(m.name, m.dtype, m.shape) for m in metadata] == [
                (name, tensor.dtype, tuple(tensor.shape)) for name, tensor in pairs
            ]
            assert [n for n, _ in self.source] == [n for n, _ in pairs]
            self.client.start_weight_update()
            self.client.update_weights({"names": [m.name for m in metadata]})
            if failure == "send":
                raise ValueError("send failed")
            received.extend((n, t.clone()) for n, t in pairs)
            self.client.finish_weight_update()

        def shutdown(self):
            events.append("shutdown")
            closed.set()

    def init(info, *, client, source):
        assert threading.get_ident() != main_thread
        assert vars(info) == dict(rank=0, master_address="127.0.0.1", master_port=54321, world_size=3, packed=False)
        client.init_weight_transfer_engine({"rank_offset": 1})
        return Trainer(client, source)

    monkeypatch.setattr(Engine, "trainer_init", init)
    config = SimpleNamespace(
        tensor_model_parallel_size=2,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=16 / (1024 * 1024)),
    )
    adapter = SimpleNamespace(
        config=config, _pd_role=None, rollout_rank=0, replica_rank=0, _get_server_name_prefix=lambda: "vllm_"
    )

    def source():
        if failure == "empty":
            return
        storage = torch.zeros(2)
        for i in range(3):
            assert threading.get_ident() == main_thread
            storage.fill_(i)
            yield str(i), storage
        if failure == "source":
            raise ValueError("source failed")
        yield "large", torch.full((20,), 7.0)

    if failure:
        pattern = "empty HCCL" if failure == "empty" else f"{failure} failed"
        with pytest.raises(ValueError, match=pattern):
            asyncio.run(native.send_ascend_hccl_weights(adapter, source()))
        assert closed.wait(2), "Failed trainer engines must be shut down"
        assert adapter._ascend_hccl_state is None
        assert "barrier" not in events
        if failure != "finish":
            assert "finish_weight_update" not in events
        return

    asyncio.run(native.send_ascend_hccl_weights(adapter, source()))
    assert [n for n, t in received] == ["0", "1", "2", "large"]
    assert [t[0].item() for n, t in received] == [0, 1, 2, 7]
    assert events[0:2] == ["init_weight_transfer_engine", "start_weight_update"]
    assert events[-2:] == ["finish_weight_update", "barrier"]
    assert events.count("update_weights") == 3
    assert events.count("start_weight_update") == 1
    assert events.count("finish_weight_update") == 1
    # The existing communicator is reused on subsequent training steps.
    asyncio.run(native.send_ascend_hccl_weights(adapter, source()))
    assert events.count("init_weight_transfer_engine") == 1
    assert events.count("start_weight_update") == 2
    assert events.count("finish_weight_update") == 2
    # Non-senders still consume every item; skipping this would deadlock FSDP.
    adapter.rollout_rank = 1
    consumed = []

    def drain():
        for i in range(3):
            consumed.append(i)
            yield str(i), torch.zeros(1)

    before = len(events)
    asyncio.run(native.send_ascend_hccl_weights(adapter, drain()))
    assert consumed == [0, 1, 2]
    assert events[before:] == ["barrier"]
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="at least two TP replicas"):
        asyncio.run(native.send_ascend_hccl_weights(adapter, source()))


@pytest.mark.parametrize("phase", ["init", "send"])
def test_hccl_cancellation_reclaims_engine_after_background_call_returns(monkeypatch, phase):
    import asyncio
    import threading

    import ray
    import torch.distributed as dist
    import vllm.utils.network_utils as network
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerWeightTransferEngine as Engine

    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Trainer:
        device = torch.device("cpu")

        def send_weights(self):
            entered.set()
            assert release.wait(3)

        def shutdown(self):
            closed.set()

    def initialize(*args, **kwargs):
        if phase == "init":
            entered.set()
            assert release.wait(3)
        return Trainer()

    monkeypatch.setattr(Engine, "trainer_init", initialize)
    monkeypatch.setattr(ray, "get_actor", lambda name: object())
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "barrier", _unexpected_dependency_call)
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "4")
    monkeypatch.setattr(network, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(network, "get_open_port", lambda: 54321)
    monkeypatch.setattr(
        native,
        "get_torch_device",
        lambda: SimpleNamespace(synchronize=lambda: None, device=lambda device: contextlib.nullcontext()),
    )
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            tensor_model_parallel_size=2,
            data_parallel_size=1,
            pipeline_model_parallel_size=1,
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1),
        ),
        _pd_role=None,
        rollout_rank=0,
        replica_rank=0,
        _get_server_name_prefix=lambda: "vllm_",
    )

    async def cancel_update():
        task = asyncio.create_task(native.send_ascend_hccl_weights(adapter, [("weight", torch.ones(2))]))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not closed.is_set(), "Do not close a communicator while its native call still uses it"
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 2)

    asyncio.run(cancel_update())
    assert adapter._ascend_hccl_state is None


@pytest.mark.parametrize("after_first_bucket", [False, True])
def test_native_receiver_abort_does_not_finalize(monkeypatch, after_first_bucket):
    """A failed source must terminate the receiver without committing partial weights."""
    events = []
    messages = []
    if after_first_bucket:
        messages.append({"update_info": {"names": ["weight"]}, "is_last": False})
    messages.append({"abort": True, "error": "source iterator failed"})
    incoming = iter(messages)

    class Socket:
        def setsockopt(self, *args):
            pass

        def connect(self, *args):
            pass

        def recv_pyobj(self):
            return next(incoming)

        def send_pyobj(self, reply):
            events.append(("ack", reply["ok"]))

        def close(self, **kwargs):
            events.append("close")

    class Worker:
        def init_weight_transfer_engine(self, info):
            events.append("init")

        def start_weight_update(self):
            events.append("start")

        def update_weights(self, info):
            events.append("update")

        def finish_weight_update(self):
            events.append("finish")

    monkeypatch.setattr(native.zmq.Context, "instance", lambda: SimpleNamespace(socket=lambda kind: Socket()))
    monkeypatch.setattr(native, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: None))
    with pytest.raises(RuntimeError, match="source iterator failed"):
        native.receive_ascend_native_weights(Worker(), "unused")
    expected = ["init", "start", "update", ("ack", True)] if after_first_bucket else []
    assert events == expected + [("ack", False), "close"]


@pytest.mark.parametrize("after_first_bucket", [False, True])
def test_reload_receiver_abort_does_not_finalize(monkeypatch, after_first_bucket):
    import torch_npu.multiprocessing.reductions as reductions
    from vllm_ascend.distributed.weight_transfer import npu_ipc_engine

    events = []
    messages = []
    if after_first_bucket:
        messages.append(
            {
                "update_info": {
                    "names": ["weight"],
                    "shapes": [[2]],
                    "dtype_names": ["float32"],
                    "ipc_handles": {"uuid": (0, 0, 0, 0, 0, 0, 0)},
                    "tensor_sizes": [8],
                },
                "is_last": False,
            }
        )
    messages.append({"abort": True, "error": "source iterator failed"})
    incoming = iter(messages)

    class Socket:
        def setsockopt(self, *args):
            pass

        def connect(self, *args):
            pass

        def recv_pyobj(self):
            return next(incoming)

        def send_pyobj(self, reply):
            events.append(("ack", reply["ok"]))

        def close(self, **kwargs):
            events.append("close")

    class Worker:
        device = torch.device("cpu:0")

        def reload_weights(self, weights_iterator, is_checkpoint_format):
            assert is_checkpoint_format
            for _ in weights_iterator:
                events.append("weight")
            events.append("finish")

    monkeypatch.setattr(native.zmq.Context, "instance", lambda: SimpleNamespace(socket=lambda kind: Socket()))
    monkeypatch.setattr(native, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(npu_ipc_engine, "npu_generate_uuid", lambda *args: "uuid")
    monkeypatch.setattr(reductions, "rebuild_npu_tensor", lambda *args: torch.tensor([1.0, 2.0]).view(torch.uint8))
    with pytest.raises(RuntimeError, match="source iterator failed"):
        native.receive_ascend_reload_weights(Worker(), "unused")
    expected = ["weight", ("ack", True)] if after_first_bucket else []
    assert events == expected + [("ack", False), "close"]


@pytest.mark.parametrize("waiting_reply, finished", [(True, False), (False, True)])
def test_sender_abort_does_not_send_when_request_is_pending_or_finished(waiting_reply, finished):
    events = []

    class Socket:
        def setsockopt(self, *args):
            events.append("setsockopt")

        def send_pyobj(self, value):
            events.append("send")

        def recv_pyobj(self):
            events.append("recv")
            return {"ok": False}

    sender = native.AscendNativeIPCWeightSender("unused")
    sender.socket = Socket()
    sender.receiver_started = True
    sender._waiting_reply = waiting_reply
    sender._transfer_finished = finished
    sender.abort(RuntimeError("source iterator failed"))
    assert events == []
    assert sender._waiting_reply is waiting_reply
    assert sender._transfer_finished is finished


@pytest.mark.parametrize("receiver_mode", ["ipc", "reload"])
def test_real_zmq_abort_terminates_receiver_thread(monkeypatch, receiver_mode):
    """Actual REQ/REP on localhost; no device or vLLM engine is required."""
    import queue
    import threading

    import torch_npu.multiprocessing.reductions as reductions
    from vllm_ascend.distributed.weight_transfer import npu_ipc_engine

    events = []
    errors = queue.Queue()

    class Worker:
        device = torch.device("cpu:0")

        def init_weight_transfer_engine(self, info):
            events.append("init")

        def start_weight_update(self):
            events.append("start")

        def update_weights(self, info):
            events.append("update")

        def finish_weight_update(self):
            events.append("finish")

        def reload_weights(self, weights_iterator, is_checkpoint_format):
            assert is_checkpoint_format
            for name, tensor in weights_iterator:
                assert name == "weight"
                assert tensor.tolist() == [1.0, 2.0]
                events.append("update")
            events.append("finish")

    monkeypatch.setattr(native, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(npu_ipc_engine, "npu_generate_uuid", lambda *args: "uuid")
    monkeypatch.setattr(reductions, "rebuild_npu_tensor", lambda *args: torch.tensor([1.0, 2.0]).view(torch.uint8))
    request = native.zmq.Context.instance().socket(native.zmq.REQ)
    request.setsockopt(native.zmq.LINGER, 0)
    request.setsockopt(native.zmq.SNDTIMEO, 3000)
    request.setsockopt(native.zmq.RCVTIMEO, 3000)
    port = request.bind_to_random_port("tcp://127.0.0.1")
    address = f"tcp://127.0.0.1:{port}"
    receive = native.receive_ascend_native_weights if receiver_mode == "ipc" else native.receive_ascend_reload_weights

    def run_receiver():
        try:
            receive(Worker(), address)
        except BaseException as exc:
            errors.put(exc)

    receiver = threading.Thread(target=run_receiver, daemon=True)
    receiver.start()
    try:
        request.send_pyobj(
            {
                "update_info": {
                    "names": ["weight"],
                    "shapes": [[2]],
                    "dtype_names": ["float32"],
                    "ipc_handles": {"uuid": (0, 0, 0, 0, 0, 0, 0)},
                    "tensor_sizes": [8],
                },
                "is_last": False,
            }
        )
        assert request.recv_pyobj() == {"ok": True}
        request.send_pyobj({"abort": True, "error": "source iterator failed"})
        acknowledgement = request.recv_pyobj()
        assert acknowledgement["ok"] is False
        assert "source iterator failed" in acknowledgement["error"]
        receiver.join(timeout=3)
        assert not receiver.is_alive(), "Receiver must exit after acknowledging abort"
        error = errors.get_nowait()
        assert isinstance(error, RuntimeError)
        assert "source iterator failed" in str(error)
        assert events.count("update") == 1
        assert "finish" not in events
    finally:
        request.close(linger=0)


def test_oversized_foreign_device_tensor_rejected_before_export(monkeypatch):
    sender, _ = _cpu_sender(monkeypatch, bucket_bytes=4)
    with pytest.raises(ValueError, match="sender buffer device"):
        sender._direct_send_large_weight("foreign", torch.empty(8, device="meta"))
