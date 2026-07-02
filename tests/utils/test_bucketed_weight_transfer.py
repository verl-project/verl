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
import uuid

import pytest
import torch

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

    def on_bucket_received(weights, *, is_last):
        del is_last
        received.extend([(name, t.clone()) for name, t in weights])

    receiver.receive_weights(on_bucket_received=on_bucket_received)
    # Only send lightweight metadata + checksum back through the queue
    summaries = [(name, t.dtype, tuple(t.shape), t.float().sum().item()) for name, t in received]
    result_queue.put(summaries)


def test_iter_weights_rebuilds_direct_sent_weight(monkeypatch):
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bucketed_weight_transfer
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    rebuilt_tensor = torch.arange(6, dtype=torch.float32).view(2, 3)

    class _FakeDevice:
        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def ipc_collect():
            pass

        @staticmethod
        def empty_cache():
            pass

    class _FakeSocket:
        def __init__(self):
            self.sent = 0

        def recv_pyobj(self):
            return {
                "bucket_meta": {
                    "large.weight": {
                        "name": "large.weight",
                        "shape": rebuilt_tensor.shape,
                        "dtype": rebuilt_tensor.dtype,
                        "offset": 0,
                        "handle": ("direct-ipc-handle",),
                    }
                },
                "is_last": True,
            }

        def send(self, payload):
            assert payload == b""
            self.sent += 1

        def close(self):
            pass

    receiver = BucketedWeightReceiver("ipc:///tmp/unused.sock", device=torch.device("cpu"), use_shm=False)
    fake_socket = _FakeSocket()

    def _init_socket():
        receiver.socket = fake_socket

    def _init_buffer():
        receiver.buffer = torch.empty(0, dtype=torch.uint8)

    monkeypatch.setattr(receiver, "_init_socket", _init_socket)
    monkeypatch.setattr(receiver, "_init_buffer", _init_buffer)
    monkeypatch.setattr(bucketed_weight_transfer, "rebuild_ipc", lambda _handle, _device_id: rebuilt_tensor)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeDevice)

    received = list(receiver.iter_weights())

    assert fake_socket.sent == 1
    assert received[0][0] == "large.weight"
    torch.testing.assert_close(received[0][1], rebuilt_tensor)


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
