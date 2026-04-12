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
"""
Bucketed weight transfer via ZMQ + IPC (or shared memory fallback).

Not recommended depending on vllm for this file.
"""

import gc
import logging
import os
from multiprocessing import shared_memory
from typing import Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


class BucketedWeightSender:
    """
    Send model weights via bucketed IPC transfer over ZMQ.

    Packs weight tensors into a fixed-size communication buffer and sends them
    in buckets to the receiver. Supports CUDA IPC and shared memory fallback.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/rl-colocate-zmq-<uuid>.sock")
        bucket_size_mb: Communication buffer size in MB
        use_shm: Use shared memory instead of CUDA IPC (for NPU compatibility)
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    async def async_send_weights(self, weights):
        """
        Send weights to the receiver. Accepts a sync generator or async iterator.

        Args:
            weights: Generator or async iterator yielding (name, tensor) pairs
        """
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            # dtype = PrecisionType.to_dtype(self.config.dtype)
            async for name, weight in ensure_async_iterator(weights):
                weight_bytes = weight.view(-1).view(torch.uint8)
                tensor_bytes = weight.nbytes
                tensor_offset = 0

                while tensor_offset < tensor_bytes:
                    chunk_bytes = min(tensor_bytes - tensor_offset, self.bucket_size - offset)
                    
                    if chunk_bytes == 0:
                        get_torch_device().synchronize()
                        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                        self.socket.recv()
                        bucket_meta = {}
                        offset = 0
                        chunk_bytes = min(tensor_bytes - tensor_offset, self.bucket_size - offset)

                    bucket_meta[name] = {
                        "name": name,
                        "shape": weight.shape,
                        "dtype": weight.dtype,
                        "tensor_offset": tensor_offset,
                        "offset": offset,
                        "chunk_bytes": chunk_bytes,
                        "is_complete": tensor_offset + chunk_bytes == tensor_bytes
                    }
                    
                    self.buffer[offset : offset + chunk_bytes].copy_(
                        weight_bytes[tensor_offset : tensor_offset + chunk_bytes], non_blocking=True
                    )
                    
                    offset += chunk_bytes
                    tensor_offset += chunk_bytes
                    
                    # If bucket is entirely full but tensor is not fully written, flush it
                    if offset == self.bucket_size and tensor_offset < tensor_bytes:
                        get_torch_device().synchronize()
                        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                        self.socket.recv()
                        bucket_meta = {}
                        offset = 0

            # send the last bucket
            get_torch_device().synchronize()
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self.socket.recv()
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REQ socket and bind."""
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        """build communication buffer"""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            # Create unique name for shared memory
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        del self.buffer
        self.buffer = None
        
        # Explicit python GC to collect any stray tensor views
        # targeting the shared memory before closing it.
        gc.collect()

        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()


class BucketedWeightReceiver:
    """
    Receive model weights via bucketed IPC transfer over ZMQ.

    Receives weight tensors from BucketedWeightSender and passes each
    bucket to a callback for processing (e.g., loading into the model).

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: Target device for received tensors
        use_shm: Use shared memory instead of CUDA IPC
    """

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def receive_weights(self, on_bucket_received: callable):
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)]) called per bucket.
        """
        try:
            self._init_socket()
            self._init_buffer()

            # receive bucket and update weights
            partial_tensors = {}
            while True:
                metadata = self.socket.recv_pyobj()
                self._process_bucket(metadata, partial_tensors, on_bucket_received)
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _process_bucket(self, metadata, partial_tensors, on_bucket_received):
        """Process a single bucket in a separate method to ensure local variables 
        (e.g., tensor views holding shm references) go out of scope immediately."""
        weights = []
        for name, meta in metadata["bucket_meta"].items():
            shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
            tensor_offset = meta.get("tensor_offset", 0)
            chunk_bytes = meta.get("chunk_bytes", dtype.itemsize * shape.numel())
            is_complete = meta.get("is_complete", True)
            
            chunk_tensor = self.buffer[offset : offset + chunk_bytes]
            full_size = dtype.itemsize * shape.numel()
            
            # Optimization: if the tensor is complete in one chunk, avoid extra allocation
            if is_complete and chunk_bytes == full_size:
                tensor = chunk_tensor.view(dtype=dtype).view(shape)
                if self.use_shm:
                    tensor = tensor.to(self.device)
                    # If device is CPU, `.to()` doesn't copy and still holds the shm buffer pointer.
                    # Force clone to detatch shared memory in local/CPU testing mode.
                    if tensor.device.type == "cpu":
                        tensor = tensor.clone()
                else: 
                    tensor = tensor.clone()

                weights.append((name, tensor))
                continue

            if name not in partial_tensors:
                # allocate an empty tensor on the target device
                partial_tensors[name] = torch.empty(shape, dtype=dtype, device=self.device)

            partial_1d = partial_tensors[name].view(-1).view(torch.uint8)
            if not self.use_shm:
                partial_1d[tensor_offset : tensor_offset + chunk_bytes].copy_(chunk_tensor, non_blocking=True)
            else:
                t = chunk_tensor.to(self.device, non_blocking=True)
                partial_1d[tensor_offset : tensor_offset + chunk_bytes].copy_(t)

            if is_complete:
                weights.append((name, partial_tensors.pop(name)))

        if weights:
            on_bucket_received(weights)
            
        get_torch_device().synchronize()
        self.socket.send(b"")

    def _init_socket(self):
        """Initialize ZMQ REP socket and connect."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        # Synchronize before releasing the buffer to ensure all async ops
        # referencing it (e.g. clone, .to()) have completed.
        get_torch_device().synchronize()
        del self.buffer
        self.buffer = None
        
        # Explicit python GC to collect any stray tensor views
        # targeting the shared memory before closing it.
        gc.collect()

        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()