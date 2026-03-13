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
def _sender_fn(zmq_handle, weight_specs, seed, bucket_size_mb, use_shm, enable_double_buffer=False):
    """Sender process: generate weights, move to device, send."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    weights = _generate_weights(weight_specs, seed)
    sender = BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=bucket_size_mb,
        use_shm=use_shm,
        enable_double_buffer=enable_double_buffer,
    )
    asyncio.run(sender.async_send_weights(iter(weights)))


def _receiver_fn(zmq_handle, use_shm, result_queue, enable_double_buffer=False):
    """Receiver process: receive weights, send back via shared memory."""
    from multiprocessing import shared_memory

    import numpy as np

    from verl.utils.device import get_device_name
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    device = torch.device(f"{get_device_name()}:0")
    receiver = BucketedWeightReceiver(
        zmq_handle=zmq_handle,
        device=device,
        use_shm=use_shm,
        enable_double_buffer=enable_double_buffer,
    )
    received = []
    receiver.receive_weights(on_bucket_received=lambda w: received.extend(w))

    # Put each tensor into shared memory to avoid Queue serialization issues
    shm_infos = []
    for name, t in received:
        t_cpu = t.cpu().contiguous()
        # Create shared memory for this tensor
        shm = shared_memory.SharedMemory(create=True, size=t_cpu.nbytes)
        # Copy tensor data to shared memory (use uint8 view to handle all dtypes including bfloat16)
        shm_buf = np.ndarray((t_cpu.nbytes,), dtype=np.uint8, buffer=shm.buf)
        shm_buf[:] = t_cpu.view(torch.uint8).flatten().numpy()
        # Send metadata: name, shm_name, shape, dtype_str
        shm_infos.append((name, shm.name, tuple(t_cpu.shape), str(t_cpu.dtype)))
        shm.close()  # Close but don't unlink - main process needs it

    result_queue.put(shm_infos)


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------
def _transfer_and_validate(weight_specs, bucket_size_mb, use_shm, enable_double_buffer=False):
    """Spawn sender + receiver processes, then validate received tensors."""
    from multiprocessing import shared_memory

    import numpy as np

    zmq_handle = _unique_zmq_handle()
    seed = 42
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    sender_p = ctx.Process(
        target=_sender_fn,
        args=(zmq_handle, weight_specs, seed, bucket_size_mb, use_shm, enable_double_buffer),
    )
    receiver_p = ctx.Process(
        target=_receiver_fn,
        args=(zmq_handle, use_shm, result_queue, enable_double_buffer),
    )

    # Start sender first (it binds), then receiver (it connects)
    sender_p.start()
    receiver_p.start()

    sender_p.join(timeout=PROCESS_TIMEOUT)
    receiver_p.join(timeout=PROCESS_TIMEOUT)

    assert sender_p.exitcode == 0, f"Sender process failed with exit code {sender_p.exitcode}"
    assert receiver_p.exitcode == 0, f"Receiver process failed with exit code {receiver_p.exitcode}"

    shm_infos = result_queue.get(timeout=5)

    # Reconstruct tensors from shared memory
    received = []
    for name, shm_name, shape, dtype_str in shm_infos:
        # dtype_str is like "torch.float32", extract the dtype name
        dtype_name = dtype_str.split(".")[-1]
        dtype = getattr(torch, dtype_name)
        # Calculate size from shape and dtype
        numel = 1
        for s in shape:
            numel *= s
        size = numel * torch.tensor([], dtype=dtype).element_size()

        shm = shared_memory.SharedMemory(name=shm_name)
        np_array = np.ndarray((size,), dtype=np.uint8, buffer=shm.buf)
        t = torch.frombuffer(np_array, dtype=dtype).reshape(shape).clone()
        received.append((name, t))
        shm.close()
        shm.unlink()  # Now we can unlink

    # Regenerate expected weights on device with the same seed, then move to CPU for comparison
    expected = [(name, t.cpu()) for name, t in _generate_weights(weight_specs, seed)]

    assert len(received) == len(expected), f"Expected {len(expected)} weights, got {len(received)}"

    for (exp_name, exp_tensor), (recv_name, recv_tensor) in zip(expected, received, strict=False):
        assert exp_name == recv_name, f"Name mismatch: expected {exp_name}, got {recv_name}"
        assert exp_tensor.shape == recv_tensor.shape, (
            f"Shape mismatch for {exp_name}: expected {exp_tensor.shape}, got {recv_tensor.shape}"
        )
        assert exp_tensor.dtype == recv_tensor.dtype, (
            f"Dtype mismatch for {exp_name}: expected {exp_tensor.dtype}, got {recv_tensor.dtype}"
        )
        assert torch.allclose(exp_tensor, recv_tensor, equal_nan=True), (
            f"Data mismatch for {exp_name}: tensors are not equal"
        )


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

    def test_large_tensor_chunked_single_weight(self):
        """Test a single tensor larger than bucket_size gets chunked correctly."""
        # 1 MB bucket; create a tensor that's ~2 MB (needs 2 chunks)
        # float32 = 4 bytes, so 2MB / 4 = 524288 elements
        numel = (2 << 20) // 4
        specs = [("large_weight", (numel,), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_large_tensor_chunked_2d(self):
        """Test a 2D tensor larger than bucket_size gets chunked correctly."""
        # 1 MB bucket; create a 2D tensor that's ~2.5 MB
        # shape (1024, 640) with float32 = 1024 * 640 * 4 = 2.5 MB
        specs = [("large_2d_weight", (1024, 640), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_large_tensor_chunked_multiple_buckets(self):
        """Test a tensor larger than bucket_size that spans multiple buckets."""
        # 1 MB bucket; create a tensor that's ~3.5 MB (needs 4 chunks)
        # float32 = 4 bytes, so 3.5MB / 4 = 917504 elements
        numel = (int(3.5 * (1 << 20))) // 4
        specs = [("huge_weight", (numel,), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_mixed_small_and_chunked_weights(self):
        """Test a mix of normal weights and chunked large weights."""
        # 1 MB bucket
        specs = [
            ("small.weight", (64, 64), torch.float32),  # ~16 KB
            ("large.weight", (1024, 512), torch.float32),  # ~2 MB (chunked)
            ("another_small.bias", (128,), torch.float32),  # ~512 bytes
            ("huge.weight", (2048, 512), torch.float32),  # ~4 MB (chunked)
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_chunked_with_different_dtypes(self):
        """Test chunked weights with different dtypes."""
        # 1 MB bucket
        specs = [
            # bf16: ~2 MB (chunked), 2 bytes per element
            ("large_bf16", (1024, 1024), torch.bfloat16),
            # float32: ~2 MB (chunked), 4 bytes per element
            ("large_fp32", (1024, 512), torch.float32),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)


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

    def test_large_tensor_chunked_single_weight(self):
        """Test a single tensor larger than bucket_size gets chunked correctly."""
        # 1 MB bucket; create a tensor that's ~2 MB (needs 2 chunks)
        # float32 = 4 bytes, so 2MB / 4 = 524288 elements
        numel = (2 << 20) // 4
        specs = [("large_weight", (numel,), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_large_tensor_chunked_2d(self):
        """Test a 2D tensor larger than bucket_size gets chunked correctly."""
        # 1 MB bucket; create a 2D tensor that's ~2.5 MB
        # shape (1024, 640) with float32 = 1024 * 640 * 4 = 2.5 MB
        specs = [("large_2d_weight", (1024, 640), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_large_tensor_chunked_multiple_buckets(self):
        """Test a tensor larger than bucket_size that spans multiple buckets."""
        # 1 MB bucket; create a tensor that's ~3.5 MB (needs 4 chunks)
        # float32 = 4 bytes, so 3.5MB / 4 = 917504 elements
        numel = (int(3.5 * (1 << 20))) // 4
        specs = [("huge_weight", (numel,), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_mixed_small_and_chunked_weights(self):
        """Test a mix of normal weights and chunked large weights."""
        # 1 MB bucket
        specs = [
            ("small.weight", (64, 64), torch.float32),  # ~16 KB
            ("large.weight", (1024, 512), torch.float32),  # ~2 MB (chunked)
            ("another_small.bias", (128,), torch.float32),  # ~512 bytes
            ("huge.weight", (2048, 512), torch.float32),  # ~4 MB (chunked)
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_chunked_with_different_dtypes(self):
        """Test chunked weights with different dtypes."""
        # 1 MB bucket
        specs = [
            # bf16: ~2 MB (chunked), 2 bytes per element
            ("large_bf16", (1024, 1024), torch.bfloat16),
            # float32: ~2 MB (chunked), 4 bytes per element
            ("large_fp32", (1024, 512), torch.float32),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)


# ---------------------------------------------------------------------------
# Double buffer tests - Shared memory
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (HAS_ACCELERATOR and not HAS_CUDA), reason="Requires shm (only tested on non-CUDA)")
class TestBucketedWeightTransferSHMDoubleBuffer:
    """Test BucketedWeightSender/Receiver with double buffer enabled via shared memory path."""

    def test_single_small_weight(self):
        specs = [("layer.weight", (32, 16), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True, enable_double_buffer=True)

    def test_multiple_buckets(self):
        # ~64 KB each x 20 = ~1.25 MB, bucket = 1 MB => spans 2 buckets
        specs = [(f"layer{i}.weight", (128, 128), torch.float32) for i in range(20)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True, enable_double_buffer=True)

    def test_large_tensor_chunked(self):
        """Test a single tensor larger than bucket_size gets chunked correctly with double buffer."""
        # 1 MB bucket; create a tensor that's ~3.5 MB (needs 4 chunks)
        numel = (int(3.5 * (1 << 20))) // 4
        specs = [("huge_weight", (numel,), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True, enable_double_buffer=True)

    def test_mixed_small_and_chunked_weights(self):
        """Test a mix of normal weights and chunked large weights with double buffer."""
        # 1 MB bucket
        specs = [
            ("small.weight", (64, 64), torch.float32),  # ~16 KB
            ("large.weight", (1024, 512), torch.float32),  # ~2 MB (chunked)
            ("another_small.bias", (128,), torch.float32),  # ~512 bytes
            ("huge.weight", (2048, 512), torch.float32),  # ~4 MB (chunked)
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True, enable_double_buffer=True)


# ---------------------------------------------------------------------------
# Double buffer tests - CUDA IPC
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not is_support_ipc(), reason="Requires IPC support")
class TestBucketedWeightTransferIPCDoubleBuffer:
    """Test BucketedWeightSender/Receiver with double buffer enabled via CUDA IPC path."""

    def test_single_small_weight(self):
        specs = [("layer.weight", (32, 16), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False, enable_double_buffer=True)

    def test_multiple_buckets(self):
        specs = [(f"layer{i}.weight", (128, 128), torch.float32) for i in range(20)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False, enable_double_buffer=True)

    def test_large_tensor_chunked(self):
        """Test a single tensor larger than bucket_size gets chunked correctly with double buffer."""
        # 1 MB bucket; create a tensor that's ~3.5 MB (needs 4 chunks)
        numel = (int(3.5 * (1 << 20))) // 4
        specs = [("huge_weight", (numel,), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False, enable_double_buffer=True)

    def test_mixed_small_and_chunked_weights(self):
        """Test a mix of normal weights and chunked large weights with double buffer."""
        # 1 MB bucket
        specs = [
            ("small.weight", (64, 64), torch.float32),  # ~16 KB
            ("large.weight", (1024, 512), torch.float32),  # ~2 MB (chunked)
            ("another_small.bias", (128,), torch.float32),  # ~512 bytes
            ("huge.weight", (2048, 512), torch.float32),  # ~4 MB (chunked)
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False, enable_double_buffer=True)
