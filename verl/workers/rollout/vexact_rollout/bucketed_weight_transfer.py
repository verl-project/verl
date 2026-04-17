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
import logging
from typing import Generator, Optional

import torch
import zmq

from verl.utils.device import get_device_name
from verl.utils.device import get_torch_device as _get_torch_device

logger = logging.getLogger(__name__)


class BucketedWeightSender:
    """
    Send model weights via bucketed IPC transfer.

    Buckets small/medium tensors together to reduce ZMQ overhead.
    Automatically chunks tensors larger than bucket size.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/weights.sock")
        bucket_size_mb: Communication buffer size in MB (default: 512)
        device: GPU device for buffer allocation
        use_shm: Use shared memory instead of CUDA IPC (for NPU compatibility)
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        device: Optional[torch.device] = None,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size = bucket_size_mb << 20
        self.device = device or torch.device(f"{get_device_name()}:0")
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """
        Send weights to receiver.

        Args:
            weights: Generator yielding (name, tensor) pairs
        """
        try:
            # Setup communication
            self._init_socket()
            self._init_buffer()

            # Send weights in buckets
            offset = 0
            bucket_meta = {}

            for name, weight in weights:
                weight_bytes = weight.nbytes

                # Case 1: Tensor fits in current bucket
                if weight_bytes <= self.bucket_size and offset + weight_bytes <= self.bucket_size:
                    self._pack_tensor(name, weight, bucket_meta, offset)
                    offset += weight_bytes

                # Case 2: Tensor fits in bucket, but current bucket full
                elif weight_bytes <= self.bucket_size:
                    if bucket_meta:
                        self._send_bucket(bucket_meta, is_last=False)
                        bucket_meta = {}
                        offset = 0

                    self._pack_tensor(name, weight, bucket_meta, offset)
                    offset += weight_bytes

                # Case 3: Tensor larger than bucket - chunk it
                else:
                    if bucket_meta:
                        self._send_bucket(bucket_meta, is_last=False)
                        bucket_meta = {}
                        offset = 0

                    self._send_chunked_tensor(name, weight)

            # Send final bucket
            if bucket_meta:
                self._send_bucket(bucket_meta, is_last=True)
            else:
                self._send_bucket({}, is_last=True)

        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ socket."""
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        """Initialize communication buffer."""
        if not self.use_shm:
            # CUDA IPC
            self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=self.device)
            from torch.multiprocessing.reductions import reduce_tensor

            handle = reduce_tensor(self.buffer)
            self.socket.send_pyobj(handle)
        else:
            # Shared memory
            import uuid
            from multiprocessing import shared_memory

            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.bucket_size)
            self.buffer = torch.frombuffer(self.shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()  # Wait for receiver ready

    def _pack_tensor(self, name: str, weight: torch.Tensor, bucket_meta: dict, offset: int):
        """Pack tensor into current bucket."""
        bucket_meta[name] = {
            "name": name,
            "shape": weight.shape,
            "dtype": weight.dtype,
            "offset": offset,
        }
        self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)

    def _send_chunked_tensor(self, name: str, weight: torch.Tensor):
        """Send large tensor in chunks."""
        weight_flat = weight.view(-1).view(torch.uint8)
        num_chunks = (weight.nbytes + self.bucket_size - 1) // self.bucket_size

        logger.info(f"Chunking tensor {name} ({weight.nbytes / (1024**2):.1f} MB) into {num_chunks} chunks")

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.bucket_size
            end = min(start + self.bucket_size, weight.nbytes)
            chunk_size = end - start

            self.buffer[:chunk_size].copy_(weight_flat[start:end], non_blocking=True)
            _get_torch_device().synchronize()

            chunk_meta = {
                name: {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": 0,
                    "chunk_info": {
                        "chunk_idx": chunk_idx,
                        "num_chunks": num_chunks,
                        "chunk_start": start,
                        "chunk_size": chunk_size,
                    },
                }
            }
            self.socket.send_pyobj({"bucket_meta": chunk_meta, "is_last": False})
            self.socket.recv()

    def _send_bucket(self, bucket_meta: dict, is_last: bool):
        """Send bucket to receiver."""
        _get_torch_device().synchronize()
        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": is_last})
        self.socket.recv()

    def _cleanup(self):
        """Clean up resources."""
        if self.socket is not None:
            self.socket.close()

        if self.buffer is not None:
            del self.buffer

        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm

        gc.collect()
        _get_torch_device().ipc_collect()
        _get_torch_device().empty_cache()


class BucketedWeightReceiver:
    """
    Receive model weights via bucketed IPC transfer.

    Receives weights from BucketedWeightSender and reassembles chunked tensors.

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: GPU device for received tensors
    """

    def __init__(
        self,
        zmq_handle: str,
        device: Optional[torch.device] = None,
    ):
        self.zmq_handle = zmq_handle
        self.device = device or torch.device(f"{get_device_name()}:0")

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None
        self.use_shm = False

    def receive_weights(self, on_bucket_received: callable):
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)]) called per bucket.
        """
        chunk_buffers = {}

        try:
            self._init_socket()
            self._init_buffer()

            # Receive buckets
            while True:
                metadata = self.socket.recv_pyobj()
                weights = []

                for name, meta in metadata["bucket_meta"].items():
                    shape = meta["shape"]
                    dtype = meta["dtype"]
                    offset = meta["offset"]

                    # Chunked tensor
                    if "chunk_info" in meta:
                        chunk_info = meta["chunk_info"]

                        # Initialize chunk buffer
                        if name not in chunk_buffers:
                            total_size = shape.numel() * dtype.itemsize
                            chunk_buffers[name] = {
                                "data": torch.empty(total_size, dtype=torch.uint8, device=self.device),
                                "shape": shape,
                                "dtype": dtype,
                                "received": set(),
                            }

                        # Copy chunk
                        start = chunk_info["chunk_start"]
                        size = chunk_info["chunk_size"]
                        end = start + size

                        tensor = self.buffer[:size]
                        if not self.use_shm:
                            tensor = tensor.clone()
                        else:
                            tensor = tensor.to(self.device)

                        chunk_buffers[name]["data"][start:end].copy_(tensor, non_blocking=True)
                        chunk_buffers[name]["received"].add(chunk_info["chunk_idx"])

                        # Check if complete
                        if len(chunk_buffers[name]["received"]) == chunk_info["num_chunks"]:
                            full_tensor = (
                                chunk_buffers[name]["data"]
                                .view(dtype=chunk_buffers[name]["dtype"])
                                .view(chunk_buffers[name]["shape"])
                            )
                            weights.append((name, full_tensor))
                            del chunk_buffers[name]

                    # Regular tensor
                    else:
                        size = dtype.itemsize * shape.numel()
                        tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)

                        if not self.use_shm:
                            tensor = tensor.clone()
                        else:
                            tensor = tensor.to(self.device)

                        weights.append((name, tensor))

                _get_torch_device().synchronize()
                self.socket.send(b"")

                # Process bucket weights immediately
                on_bucket_received(weights)

                if metadata["is_last"]:
                    break

        finally:
            self._cleanup()
            del chunk_buffers
            _get_torch_device().empty_cache()

    def _init_socket(self):
        """Initialize ZMQ socket."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        """Initialize communication buffer."""
        comm_metadata = self.socket.recv_pyobj()

        if isinstance(comm_metadata, tuple):
            # CUDA IPC handle
            func, args = comm_metadata
            list_args = list(args)
            list_args[6] = self.device.index  # Override device ID
            self.buffer = func(*list_args)
            self.use_shm = False
        else:
            # Shared memory
            from multiprocessing import shared_memory

            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.buffer = torch.frombuffer(self.shm.buf[:shm_size], dtype=torch.uint8)
            self.use_shm = True

        self.socket.send(b"")  # Signal ready

    def _cleanup(self):
        """Clean up resources."""
        if self.socket is not None:
            self.socket.close()

        if self.buffer is not None:
            del self.buffer

        if self.shm is not None:
            self.shm.close()
            del self.shm

        gc.collect()
        _get_torch_device().ipc_collect()
        _get_torch_device().empty_cache()
