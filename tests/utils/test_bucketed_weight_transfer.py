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
import multiprocessing as mp
import os
import threading
import uuid

import pytest
import torch
import zmq

from verl.utils.device import get_device_name, get_torch_device, is_support_ipc

PROCESS_TIMEOUT = 60

# Use string checks to avoid initializing CUDA in the main pytest process,
# which would make subsequent fork-based multiprocessing in other tests unsafe.
HAS_ACCELERATOR = get_device_name() != "cpu"
HAS_CUDA = "cuda" in get_device_name()


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
    def __init__(self, response=b"", poll_result=zmq.POLLIN):
        self.messages = []
        self.response = response
        self.poll_result = poll_result

    def send_pyobj(self, message):
        self.messages.append(message)

    def send(self, message):
        self.messages.append(message)

    def recv(self):
        return self.response

    def poll(self, _timeout, _flags):
        return self.poll_result


class _FakeTorchDevice:
    def synchronize(self):
        pass

    def empty_cache(self):
        pass

    def ipc_collect(self):
        pass


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


def test_receiver_callback_failure_reaches_sender(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    response = bucketed_weight_transfer._encode_receiver_error(KeyError("embed_tokens.weight"))
    socket = _FakeSocket(response=response)
    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle="ipc:///tmp/test-bwt-unused.sock",
        bucket_size_mb=1,
        use_shm=True,
    )

    monkeypatch.setattr(sender, "_init_socket", lambda: setattr(sender, "socket", socket))
    monkeypatch.setattr(sender, "_init_buffer", lambda: setattr(sender, "buffer", torch.empty(0)))
    monkeypatch.setattr(sender, "_cleanup", lambda: None)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    with pytest.raises(RuntimeError, match=r"KeyError: 'embed_tokens\.weight'"):
        asyncio.run(sender.async_send_weights(iter(())))


def test_sender_times_out_without_receiver_ack(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    socket = _FakeSocket(poll_result=0)
    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle="ipc:///tmp/test-bwt-unused.sock",
        bucket_size_mb=1,
        use_shm=True,
        ack_timeout_ms=1,
    )
    monkeypatch.setattr(sender, "_init_socket", lambda: setattr(sender, "socket", socket))
    monkeypatch.setattr(sender, "_init_buffer", lambda: setattr(sender, "buffer", torch.empty(0)))
    monkeypatch.setattr(sender, "_cleanup", lambda: None)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    with pytest.raises(RuntimeError, match="timed out waiting 1ms"):
        asyncio.run(sender.async_send_weights(iter(())))


@pytest.mark.parametrize("phase", ["intermediate", "oversized"])
def test_sender_propagates_receiver_error_for_every_bucket_send(monkeypatch, phase):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    socket = _FakeSocket(response=bucketed_weight_transfer._encode_receiver_error(ValueError(phase)))
    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle="ipc:///tmp/test-bwt-unused.sock",
        bucket_size_mb=1,
        use_shm=False,
    )
    monkeypatch.setattr(sender, "_init_socket", lambda: setattr(sender, "socket", socket))
    monkeypatch.setattr(sender, "_init_buffer", lambda: setattr(sender, "buffer", torch.empty(1 << 20, dtype=torch.uint8)))
    monkeypatch.setattr(sender, "_cleanup", lambda: None)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    if phase == "intermediate":
        weights = [("a", torch.empty(200_000, dtype=torch.uint8)), ("b", torch.empty(200_000, dtype=torch.uint8))]
    else:
        weights = [("large", torch.empty(300_000))]

    with pytest.raises(RuntimeError, match=f"ValueError: {phase}"):
        asyncio.run(sender.async_send_weights(iter(weights)))


def _receiver_with_socket(monkeypatch, metadata, *, use_shm=False):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    class ReceiverSocket(_FakeSocket):
        def recv_pyobj(self):
            return metadata

    socket = ReceiverSocket()
    receiver = bucketed_weight_transfer.BucketedWeightReceiver(
        zmq_handle="ipc:///tmp/test-bwt-unused.sock",
        device=torch.device("cpu"),
        use_shm=use_shm,
    )
    monkeypatch.setattr(receiver, "_init_socket", lambda: setattr(receiver, "socket", socket))
    monkeypatch.setattr(receiver, "_init_buffer", lambda: None)
    monkeypatch.setattr(receiver, "_cleanup", lambda: None)
    return receiver, socket


def test_receiver_reports_metadata_decode_failure(monkeypatch):
    receiver, socket = _receiver_with_socket(monkeypatch, {"is_last": True})

    with pytest.raises(KeyError, match="bucket_meta"):
        receiver.receive_weights(lambda *_: None)

    assert socket.messages == [b"error:KeyError: 'bucket_meta'"]


def test_receiver_reports_ipc_rebuild_failure(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    receiver, socket = _receiver_with_socket(
        monkeypatch,
        {
            "bucket_meta": {
                "weight": {"shape": torch.Size([1]), "dtype": torch.float32, "offset": 0, "handle": ("bad", ())}
            },
            "is_last": True,
        },
    )
    monkeypatch.setattr(bucketed_weight_transfer, "rebuild_ipc", lambda *_: (_ for _ in ()).throw(ValueError("bad IPC")))

    with pytest.raises(ValueError, match="bad IPC"):
        receiver.receive_weights(lambda *_: None)

    assert socket.messages == [b"error:ValueError: bad IPC"]


def test_receiver_reports_tensor_conversion_failure(monkeypatch):
    class FailingTensor:
        def view(self, *_args, **_kwargs):
            return self

        def to(self, _device):
            raise RuntimeError("tensor conversion failed")

    class FailingBuffer:
        def __getitem__(self, _index):
            return FailingTensor()

    receiver, socket = _receiver_with_socket(
        monkeypatch,
        {
            "bucket_meta": {
                "weight": {"shape": torch.Size([1]), "dtype": torch.float32, "offset": 0, "handle": None}
            },
            "is_last": True,
        },
        use_shm=True,
    )
    receiver.buffer = FailingBuffer()

    with pytest.raises(RuntimeError, match="tensor conversion failed"):
        receiver.receive_weights(lambda *_: None)

    assert socket.messages == [b"error:RuntimeError: tensor conversion failed"]


def test_receiver_reports_synchronize_failure(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    receiver, socket = _receiver_with_socket(monkeypatch, {"bucket_meta": {}, "is_last": True})
    monkeypatch.setattr(
        bucketed_weight_transfer,
        "get_torch_device",
        lambda: type("FailingDevice", (), {"synchronize": lambda self: (_ for _ in ()).throw(RuntimeError("sync failed"))})(),
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        receiver.receive_weights(lambda *_: None)

    assert socket.messages == [b"error:RuntimeError: sync failed"]


def _receiver_hard_exit_after_buffer_ack(zmq_handle):
    socket = zmq.Context().socket(zmq.REP)
    socket.connect(zmq_handle)
    socket.recv_pyobj()
    socket.send(b"")
    socket.recv_pyobj()
    os._exit(0)


def test_sender_times_out_after_receiver_hard_exit(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    zmq_handle = _unique_zmq_handle()
    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=1,
        use_shm=True,
        ack_timeout_ms=1_000,
    )
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())
    sender_error = []

    def send():
        try:
            asyncio.run(sender.async_send_weights(iter(())))
        except Exception as exc:
            sender_error.append(exc)

    sender_thread = threading.Thread(target=send)
    receiver_process = mp.get_context("spawn").Process(target=_receiver_hard_exit_after_buffer_ack, args=(zmq_handle,))
    sender_thread.start()
    receiver_process.start()
    receiver_process.join(timeout=5)
    sender_thread.join(timeout=5)

    assert receiver_process.exitcode == 0
    assert not sender_thread.is_alive()
    assert len(sender_error) == 1
    assert "timed out waiting 1000ms" in str(sender_error[0])


def test_receiver_reports_callback_failure_before_reraising(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    class _ReceiverSocket(_FakeSocket):
        def recv_pyobj(self):
            return {"bucket_meta": {}, "is_last": True}

    def fail_weight_load(_weights, _is_last):
        raise KeyError("embed_tokens.weight")

    socket = _ReceiverSocket()
    receiver = bucketed_weight_transfer.BucketedWeightReceiver(
        zmq_handle="ipc:///tmp/test-bwt-unused.sock",
        device=torch.device("cpu"),
        use_shm=True,
    )
    monkeypatch.setattr(receiver, "_init_socket", lambda: setattr(receiver, "socket", socket))
    monkeypatch.setattr(receiver, "_init_buffer", lambda: None)
    monkeypatch.setattr(receiver, "_cleanup", lambda: None)

    with pytest.raises(KeyError, match="embed_tokens.weight"):
        receiver.receive_weights(fail_weight_load)

    assert socket.messages == [bucketed_weight_transfer._encode_receiver_error(KeyError("embed_tokens.weight"))]


def test_receiver_failure_reaches_sender_over_zmq(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())
    zmq_handle = _unique_zmq_handle()
    sender = bucketed_weight_transfer.BucketedWeightSender(zmq_handle=zmq_handle, bucket_size_mb=1, use_shm=True)
    receiver = bucketed_weight_transfer.BucketedWeightReceiver(
        zmq_handle=zmq_handle,
        device=torch.device("cpu"),
        use_shm=True,
    )
    sender_error = []

    def fail_weight_load(_weights, _is_last):
        raise KeyError("embed_tokens.weight")

    def send():
        try:
            asyncio.run(sender.async_send_weights(iter(())))
        except Exception as exc:
            sender_error.append(exc)

    sender_thread = threading.Thread(target=send)
    sender_thread.start()
    with pytest.raises(KeyError, match="embed_tokens.weight"):
        receiver.receive_weights(fail_weight_load)
    sender_thread.join(timeout=5)

    assert not sender_thread.is_alive()
    assert len(sender_error) == 1
    assert isinstance(sender_error[0], RuntimeError)
    assert "KeyError: 'embed_tokens.weight'" in str(sender_error[0])


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
