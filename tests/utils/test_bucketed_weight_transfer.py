# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Tests for BucketedWeightSender and BucketedWeightReceiver.

Sender and receiver run in separate processes to match real-world usage
and because CUDA IPC requires distinct processes.
"""

import asyncio
import gc
import importlib.util
import multiprocessing as mp
import uuid
import weakref
from pathlib import Path

import pytest
import torch

from verl.utils.device import get_device_name, get_torch_device, is_support_ipc

PROCESS_TIMEOUT = 60

# Use string checks to avoid initializing CUDA in the main pytest process,
# which would make subsequent fork-based multiprocessing in other tests unsafe.
HAS_ACCELERATOR = get_device_name() != "cpu"
HAS_CUDA = "cuda" in get_device_name()


def _load_bucketed_weight_transfer():
    module_path = Path(__file__).resolve().parents[2] / "verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"
    spec = importlib.util.spec_from_file_location("bucketed_weight_transfer_iterator_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unique_zmq_handle():
    return f"ipc:///tmp/test-bwt-{uuid.uuid4().hex}.sock"


def _generate_weights(weight_specs, seed):
    """Deterministically generate weights on the best available device from specs.

    Args:
        weight_specs: list of (name, shape, dtype) tuples
        seed: random seed for reproducibility
    Returns:
        list of (name, tensor_on_device) tuples
    """
    device_name = get_device_name()
    device = torch.device(f"{device_name}:0")
    get_torch_device().manual_seed(seed)
    weights = []
    for name, shape, dtype in weight_specs:
        # Generate in float32 then cast, since torch.randn doesn't support all dtypes
        t = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
        weights.append((name, t))
    return weights


class _FakeSocket:
    def __init__(self):
        self.messages = []

    def send_pyobj(self, message):
        self.messages.append(message)

    def recv(self):
        return b""


class _FakeTorchDevice:
    def synchronize(self):
        pass


class _ScriptedReceiverSocket:
    def __init__(self, payloads, buffer, replacement):
        self.payloads = iter(payloads)
        self.buffer = buffer
        self.replacement = replacement
        self.acks = 0

    def recv_pyobj(self):
        return next(self.payloads)

    def send(self, payload):
        assert payload == b""
        self.acks += 1
        if self.acks == 1:
            self.buffer.copy_(self.replacement.view(torch.uint8))


def test_sender_accepts_strided_tensor(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    base = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    weight = base[:, 0, :]
    buffer = torch.empty(weight.nbytes, dtype=torch.uint8)
    socket = _FakeSocket()
    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle="ipc:///tmp/test-bwt-unused.sock",
        bucket_size_mb=1,
        use_shm=True,
    )

    assert not weight.is_contiguous()
    with pytest.raises(RuntimeError):
        weight.view(-1).view(torch.uint8)

    monkeypatch.setattr(sender, "_init_socket", lambda: setattr(sender, "socket", socket))
    monkeypatch.setattr(sender, "_init_buffer", lambda: setattr(sender, "buffer", buffer))
    monkeypatch.setattr(sender, "_cleanup", lambda: None)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    asyncio.run(sender.async_send_weights(iter([("strided", weight)])))

    recovered = buffer.view(dtype=weight.dtype).view(weight.shape)

    assert socket.messages == [
        {
            "bucket_meta": {
                "strided": {
                    "name": "strided",
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": 0,
                    "handle": None,
                }
            },
            "is_last": True,
        }
    ]
    assert buffer.dtype == torch.uint8
    assert buffer.numel() == weight.nbytes
    assert torch.equal(recovered, weight)


def _receiver_metadata(name, tensor, is_last):
    return {
        "bucket_meta": {
            name: {
                "name": name,
                "shape": tensor.shape,
                "dtype": tensor.dtype,
                "offset": 0,
                "handle": None,
            }
        },
        "is_last": is_last,
    }


def test_receiver_iterator_owns_tensors_before_bucket_ack(monkeypatch):
    bucketed_weight_transfer = _load_bucketed_weight_transfer()

    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([9.0, 10.0])
    buffer = first.clone().view(torch.uint8)
    socket = _ScriptedReceiverSocket(
        [_receiver_metadata("first", first, False), _receiver_metadata("second", second, True)],
        buffer,
        second,
    )
    receiver = bucketed_weight_transfer.BucketedWeightReceiver("unused", torch.device("cpu"))

    monkeypatch.setattr(receiver, "_init_socket", lambda: setattr(receiver, "socket", socket))
    monkeypatch.setattr(receiver, "_init_buffer", lambda: setattr(receiver, "buffer", buffer))
    monkeypatch.setattr(receiver, "_cleanup", lambda: None)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    received = list(receiver.iter_weights(own_tensors=True))

    assert socket.acks == 2
    torch.testing.assert_close(received[0][1], first)
    torch.testing.assert_close(received[1][1], second)


def test_receiver_can_defer_final_ack_until_reload_finalize(monkeypatch):
    bucketed_weight_transfer = _load_bucketed_weight_transfer()

    final = torch.tensor([3.0])
    buffer = final.clone().view(torch.uint8)
    socket = _ScriptedReceiverSocket([_receiver_metadata("final", final, True)], buffer, final)
    receiver = bucketed_weight_transfer.BucketedWeightReceiver("unused", torch.device("cpu"))

    monkeypatch.setattr(receiver, "_init_socket", lambda: setattr(receiver, "socket", socket))
    monkeypatch.setattr(receiver, "_init_buffer", lambda: setattr(receiver, "buffer", buffer))
    monkeypatch.setattr(receiver, "_cleanup", lambda: None)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    received = list(receiver.iter_weights(own_tensors=True, defer_last_ack=True))
    assert received[0][0] == "final"
    assert socket.acks == 0
    assert receiver.iterator_exhausted is True

    receiver.complete_deferred_last_ack()
    assert socket.acks == 1


def test_receiver_iterator_close_drains_sender(monkeypatch):
    bucketed_weight_transfer = _load_bucketed_weight_transfer()

    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    buffer = first.clone().view(torch.uint8)
    socket = _ScriptedReceiverSocket(
        [_receiver_metadata("first", first, False), _receiver_metadata("second", second, True)],
        buffer,
        second,
    )
    receiver = bucketed_weight_transfer.BucketedWeightReceiver("unused", torch.device("cpu"))

    monkeypatch.setattr(receiver, "_init_socket", lambda: setattr(receiver, "socket", socket))
    monkeypatch.setattr(receiver, "_init_buffer", lambda: setattr(receiver, "buffer", buffer))
    monkeypatch.setattr(receiver, "_cleanup", lambda: None)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    iterator = receiver.iter_weights(own_tensors=True)
    assert next(iterator)[0] == "first"
    iterator.close()

    assert socket.acks == 2


def test_receiver_never_consumed_iterator_drains_sender(monkeypatch):
    bucketed_weight_transfer = _load_bucketed_weight_transfer()

    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    buffer = first.clone().view(torch.uint8)
    socket = _ScriptedReceiverSocket(
        [_receiver_metadata("first", first, False), _receiver_metadata("second", second, True)],
        buffer,
        second,
    )
    receiver = bucketed_weight_transfer.BucketedWeightReceiver("unused", torch.device("cpu"))

    monkeypatch.setattr(receiver, "_init_socket", lambda: setattr(receiver, "socket", socket))
    monkeypatch.setattr(receiver, "_init_buffer", lambda: setattr(receiver, "buffer", buffer))
    monkeypatch.setattr(receiver, "_cleanup", lambda: None)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    iterator = receiver.iter_weights(own_tensors=True)
    receiver.close_weight_iterator(iterator)

    assert socket.acks == 2


def _cleanup_order_fixture(monkeypatch, *, use_shm=False, single_bucket=False):
    module = _load_bucketed_weight_transfer()
    receiver = module.BucketedWeightReceiver("unused", torch.device("cpu"), use_shm=use_shm)
    events, aliases = [], []
    tensor = torch.tensor([1.0])
    payloads = [_receiver_metadata("last", tensor, True)]
    if not single_bucket:
        payloads.insert(0, _receiver_metadata("first", tensor, False))

    class Device:
        def synchronize(self):
            events.append("fence")

        def ipc_collect(self):
            events.append("ipc_collect")

        def empty_cache(self):
            events.append("empty_cache")

    class SharedMemory:
        closed = False

        def close(self):
            assert receiver.buffer is None
            self.closed = True
            events.append("shm_close")

    shm = SharedMemory() if use_shm else None

    class Socket:
        def __init__(self):
            self.payloads = iter(payloads)
            self.is_last = None
            self.acks = []
            self.closed = False

        def recv_pyobj(self):
            metadata = next(self.payloads)
            self.is_last = metadata["is_last"]
            return metadata

        def send(self, payload):
            assert payload == b""
            assert all(reference() is None for reference in aliases), "temporary views survive ACK"
            if self.is_last:
                assert receiver.buffer is None, "final ACK precedes mapping release"
                assert receiver.shm is None
                assert shm is None or shm.closed
            self.acks.append(self.is_last)
            events.append("ack_last" if self.is_last else "ack_more")

        def close(self):
            assert not self.closed
            self.closed = True
            events.append("socket_close")

    socket = Socket()

    def init_buffer():
        receiver.buffer = tensor.clone().view(torch.uint8)
        receiver.shm = shm

    def no_full_gc(*args, **kwargs):
        raise AssertionError("weight transfer must not rely on full GC")

    monkeypatch.setattr(receiver, "_init_socket", lambda: setattr(receiver, "socket", socket))
    monkeypatch.setattr(receiver, "_init_buffer", init_buffer)
    monkeypatch.setattr(module, "get_torch_device", lambda: Device())
    monkeypatch.setattr(module, "is_support_ipc", lambda: not use_shm)
    monkeypatch.setattr(gc, "collect", no_full_gc)
    return receiver, socket, events, aliases


@pytest.mark.parametrize("use_shm", [False, True])
@pytest.mark.parametrize("fail_on", [None, "first", "last"])
def test_callback_releases_views_and_mapping_before_ack(monkeypatch, use_shm, fail_on):
    receiver, socket, events, aliases = _cleanup_order_fixture(monkeypatch, use_shm=use_shm)

    def callback(weights, is_last):
        # Exception tracebacks may own consumer-local tensors until the caller
        # handles the error. Only the success path can require those references
        # to disappear; failure paths must still drain and unblock the sender.
        if fail_on is None:
            aliases.extend(weakref.ref(tensor) for _, tensor in weights)
        if weights[0][0] == fail_on:
            raise ValueError("consumer failed")

    if fail_on is None:
        receiver.receive_weights(callback)
    else:
        with pytest.raises(ValueError, match="consumer failed"):
            receiver.receive_weights(callback)
    assert socket.acks == [False, True]
    assert receiver.buffer is receiver.shm is receiver.socket is None
    assert events.index("ack_last") < events.index("socket_close")


@pytest.mark.parametrize("use_shm", [False, True])
@pytest.mark.parametrize("defer", [False, True])
def test_iterator_releases_mapping_after_finalize_before_final_ack(monkeypatch, use_shm, defer):
    receiver, socket, events, _ = _cleanup_order_fixture(monkeypatch, use_shm=use_shm)
    owned = list(receiver.iter_weights(own_tensors=True, defer_last_ack=defer))
    assert receiver.iterator_exhausted
    assert [name for name, _ in owned] == ["first", "last"]
    if defer:
        assert socket.acks == [False]
        assert receiver.buffer is not None
        events.append("reload_finalized")
        receiver.complete_deferred_last_ack()
        assert events.index("reload_finalized") < events.index("ack_last")
    assert socket.acks == [False, True]
    assert receiver.buffer is receiver.shm is receiver.socket is None
    for _, tensor in owned:
        torch.testing.assert_close(tensor, torch.tensor([1.0]))


@pytest.mark.parametrize("use_shm", [False, True])
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("stop", ["never_started", "first_yield", "last_yield"])
def test_abandoned_iterator_releases_mapping_and_drains_sender(monkeypatch, use_shm, defer, stop):
    receiver, socket, _, _ = _cleanup_order_fixture(monkeypatch, use_shm=use_shm, single_bucket=stop == "last_yield")
    iterator = receiver.iter_weights(own_tensors=True, defer_last_ack=defer)
    if stop != "never_started":
        assert next(iterator)[1].item() == 1.0
    receiver.close_weight_iterator(iterator)
    assert socket.acks == ([True] if stop == "last_yield" else [False, True])
    assert receiver.buffer is receiver.shm is receiver.socket is None
    assert not receiver._last_ack_deferred


def test_final_ack_still_unblocks_sender_when_mapping_cleanup_fails(monkeypatch):
    module = _load_bucketed_weight_transfer()
    receiver = module.BucketedWeightReceiver("unused", torch.device("cpu"))
    events = []

    class Socket:
        def send(self, payload):
            assert payload == b""
            events.append("ack")

        def close(self):
            events.append("close")

    def fail_release():
        events.append("release_attempt")
        raise RuntimeError("mapping release failed")

    receiver.socket = Socket()
    receiver._last_ack_deferred = True
    monkeypatch.setattr(receiver, "_release_buffer", fail_release)
    with pytest.raises(RuntimeError, match="mapping release failed"):
        receiver.complete_deferred_last_ack()
    assert events == ["release_attempt", "ack", "release_attempt", "close"]
    assert receiver.socket is None and not receiver._last_ack_deferred


# ---------------------------------------------------------------------------
# Process entry points (must be module-level for pickling with spawn)
# ---------------------------------------------------------------------------
def _sender_fn(zmq_handle, weight_specs, seed, bucket_size_mb, use_shm):
    """Sender process: generate weights, move to device, send."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    weights = _generate_weights(weight_specs, seed)
    sender = BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=bucket_size_mb,
        use_shm=use_shm,
    )
    asyncio.run(sender.async_send_weights(iter(weights)))


def _receiver_fn(zmq_handle, use_shm, result_queue):
    """Receiver process: receive weights, send back (name, dtype, shape, checksum)."""
    from verl.utils.device import get_device_name
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    device = torch.device(f"{get_device_name()}:0")
    receiver = BucketedWeightReceiver(
        zmq_handle=zmq_handle,
        device=device,
        use_shm=use_shm,
    )
    received = []
    receiver.receive_weights(
        on_bucket_received=lambda w, is_last: received.extend([(name, t.clone()) for name, t in w])
    )
    # Only send lightweight metadata + checksum back through the queue
    summaries = [(name, t.dtype, tuple(t.shape), t.float().sum().item()) for name, t in received]
    result_queue.put(summaries)


def _sender_memory_rounds_fn(handles, result_queue):
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    device = get_torch_device()
    weights = _generate_weights([("weight", (1024, 1024), torch.float32)], 42)
    device.synchronize()
    baseline = device.memory_allocated()
    deltas = []
    for handle in handles:
        sender = BucketedWeightSender(handle, bucket_size_mb=16, use_shm=False)
        asyncio.run(sender.async_send_weights(iter(weights)))
        device.synchronize()
        delta = device.memory_allocated() - baseline
        # Permit small runtime bookkeeping, never an outstanding 16 MiB bucket.
        assert delta < 1024 * 1024, f"IPC bucket not reclaimed in its own round: {delta} bytes"
        deltas.append(delta)
    result_queue.put(deltas)


def _receiver_memory_rounds_fn(handles, result_queue):
    for handle in handles:
        _receiver_fn(handle, False, result_queue)


@pytest.mark.skipif(not HAS_CUDA, reason="Requires real CUDA IPC memory accounting")
def test_ipc_bucket_reclaimed_before_next_round():
    handles = [_unique_zmq_handle() for _ in range(3)]
    ctx = mp.get_context("spawn")
    sender_results, receiver_results = ctx.Queue(), ctx.Queue()
    sender = ctx.Process(target=_sender_memory_rounds_fn, args=(handles, sender_results))
    receiver = ctx.Process(target=_receiver_memory_rounds_fn, args=(handles, receiver_results))
    try:
        sender.start()
        receiver.start()
        sender.join(timeout=PROCESS_TIMEOUT)
        receiver.join(timeout=PROCESS_TIMEOUT)
        assert sender.exitcode == 0, f"sender exit={sender.exitcode}"
        assert receiver.exitcode == 0, f"receiver exit={receiver.exitcode}"
        deltas = sender_results.get(timeout=5)
        assert len(deltas) == 3 and max(deltas) < 1024 * 1024
        for _ in handles:
            summary = receiver_results.get(timeout=5)
            assert len(summary) == 1 and summary[0][0] == "weight"
    finally:
        for process in (sender, receiver):
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)
        sender_results.close()
        receiver_results.close()


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------
def _transfer_and_validate(weight_specs, bucket_size_mb, use_shm):
    """Spawn sender + receiver processes, then validate received tensors."""
    zmq_handle = _unique_zmq_handle()
    seed = 42
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    sender_p = ctx.Process(
        target=_sender_fn,
        args=(zmq_handle, weight_specs, seed, bucket_size_mb, use_shm),
    )
    receiver_p = ctx.Process(
        target=_receiver_fn,
        args=(zmq_handle, use_shm, result_queue),
    )

    # Start sender first (it binds), then receiver (it connects)
    sender_p.start()
    receiver_p.start()

    sender_p.join(timeout=PROCESS_TIMEOUT)
    receiver_p.join(timeout=PROCESS_TIMEOUT)

    assert sender_p.exitcode == 0, f"Sender process failed with exit code {sender_p.exitcode}"
    assert receiver_p.exitcode == 0, f"Receiver process failed with exit code {receiver_p.exitcode}"

    summaries = result_queue.get(timeout=5)

    # Regenerate expected weights on device with the same seed
    expected = _generate_weights(weight_specs, seed)

    assert len(summaries) == len(expected), f"Expected {len(expected)} weights, got {len(summaries)}"

    for (exp_name, exp_tensor), (recv_name, recv_dtype, recv_shape, recv_cksum) in zip(
        expected, summaries, strict=False
    ):
        assert exp_name == recv_name, f"Name mismatch: expected {exp_name}, got {recv_name}"
        assert tuple(exp_tensor.shape) == recv_shape, (
            f"Shape mismatch for {exp_name}: expected {tuple(exp_tensor.shape)}, got {recv_shape}"
        )
        assert exp_tensor.dtype == recv_dtype, (
            f"Dtype mismatch for {exp_name}: expected {exp_tensor.dtype}, got {recv_dtype}"
        )
        exp_sum = exp_tensor.float().sum().item()
        assert exp_sum == recv_cksum, f"Data mismatch for {exp_name}"


# ---------------------------------------------------------------------------
# Shared memory tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_CUDA, reason="Requires CUDA consumer for shared-memory copy regression")
def test_shared_memory_cuda_consumer_releases_buffer():
    specs = [(f"layer{i}.weight", (128, 128), torch.float32) for i in range(20)]
    _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)


@pytest.mark.skipif(not (HAS_ACCELERATOR and not HAS_CUDA), reason="Requires (shm only tested)")
class TestBucketedWeightTransferSHM:
    """Test BucketedWeightSender/Receiver via shared memory path."""

    def test_single_small_weight(self):
        specs = [("layer.weight", (32, 16), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_multiple_weights_single_bucket(self):
        specs = [
            ("layer0.weight", (16, 16), torch.float32),
            ("layer0.bias", (16,), torch.float32),
            ("layer1.weight", (16, 8), torch.bfloat16),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_multiple_buckets(self):
        # ~64 KB each x 20 = ~1.25 MB, bucket = 1 MB => spans 2 buckets
        specs = [(f"layer{i}.weight", (128, 128), torch.float32) for i in range(20)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_mixed_dtypes(self):
        specs = [
            ("fp32_param", (64, 64), torch.float32),
            ("bf16_param", (64, 64), torch.bfloat16),
            ("fp16_param", (32, 32), torch.float16),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_empty_weights(self):
        _transfer_and_validate([], bucket_size_mb=1, use_shm=True)


# ---------------------------------------------------------------------------
# CUDA IPC tests (CUDA only — IPC is not supported on NPU)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not is_support_ipc(), reason="Requires IPC support")
class TestBucketedWeightTransferIPC:
    """Test BucketedWeightSender/Receiver via CUDA IPC path."""

    def test_single_small_weight(self):
        specs = [("layer.weight", (32, 16), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_multiple_weights_single_bucket(self):
        specs = [
            ("layer0.weight", (16, 16), torch.float32),
            ("layer0.bias", (16,), torch.float32),
            ("layer1.weight", (16, 8), torch.bfloat16),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_multiple_buckets(self):
        specs = [(f"layer{i}.weight", (128, 128), torch.float32) for i in range(20)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_mixed_dtypes(self):
        specs = [
            ("__delta_spec__", (5,), torch.uint8),
            ("__positions__", (8,), torch.uint8),
            ("__values__", (2,), torch.bfloat16),
            ("fp32_param", (64, 64), torch.float32),
            ("bf16_param", (64, 64), torch.bfloat16),
            ("fp16_param", (32, 32), torch.float16),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_empty_weights(self):
        _transfer_and_validate([], bucket_size_mb=1, use_shm=False)

    def test_exact_bucket_boundary(self):
        # 1 MB bucket = 1048576 bytes; float32 = 4 bytes => 262144 elements
        numel = (1 << 20) // 4
        specs = [("exact_fit", (numel,), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_large_weight(self):
        specs = [("embedding", (1024, 1024), torch.float32)]  # 4MB
        specs.extend([(f"layer{i}.weight", (128,), torch.bfloat16) for i in range(5)])
        specs.append(("gate_up_proj", (1024, 1024), torch.float32))  # 4MB
        specs.extend([(f"layer{i}.weight", (128,), torch.bfloat16) for i in range(20)])
        specs.append(("lm_head", (1024, 1024), torch.float32))  # 4MB

        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)
