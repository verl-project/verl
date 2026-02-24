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
import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Generator, TypedDict

import ray
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.ray_utils import auto_await
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout import BaseRollout, RolloutReplica, get_rollout_class

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TensorMeta(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


class CheckpointEngineRegistry:
    """Checkpoint engine registry."""

    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):
        """Register a checkpoint engine.

        Args:
            backend: The backend of the checkpoint engine.
        """

        def wrapper(cls: type["CheckpointEngine"]):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls

        return wrapper

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        """Get the checkpoint engine class.

        Args:
            backend: The backend of the checkpoint engine.

        Returns:
            The checkpoint engine class.
        """
        return cls._registry[backend]

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        """Create a new checkpoint engine instance.

        Args:
            backend: The backend of the checkpoint engine.
            *args: Variable length argument pass to the checkpoint engine constructor.
            **kwargs: Arbitrary keyword arguments pass to the checkpoint engine constructor.

        Returns:
            A new checkpoint engine instance.
        """
        if backend not in cls._registry:
            raise ValueError(f"Checkpoint engine {backend} not registered")
        return cls._registry[backend](*args, **kwargs)


class CheckpointEngine(ABC):
    """CheckpointEngine is an abstraction to transfer weights from trainer to rollout.

    In trainer process:
    >>> trainer = EngineRegistry.new(...) # FSDP, Megatron, VeOmini, TorchTitan, ...
    >>> engine = CheckpointEngine.new(...) # NCCLCheckpointEngine, NIXLCheckpointEngine, ...
    >>> await engine.send_weights(trainer.get_per_tensor_param())

    In rollout process:
    >>> engine = CheckpointEngine.new(...)
    >>> server_adapter = ServerAdapter()
    >>> await server_adapter.update_weights(engine.get_weights()) # update weights via cuda ipc
    """

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Prepare checkpoint engine before each step send_weights/receive_weights.

        1. Allocate weight bucket.
        2. [Optional] Register weight bucket for RDMA.
        3. Return metadata to build communication topology: master ip:port, register RDMA description, etc.

        Args:
            worker_group: The worker group that the checkpoint engine will be used.

        Returns:
            A dictionary that contains the metadata of the worker group.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_topology(
        cls, trainer_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build communication topology between all workers.

        Args:
            trainer_world_size: The world size of the trainer worker group.
            rollout_world_size: The world size of the rollout replica.
            metadata: A list of metadata `prepare` from all workers.

        Returns:
            A tuple of two dictionaries that contains the communication topology for trainer and rollout worker group.
            Each dict value should be a list argument equal to the world size of the worker group to dispatch to
            `init_process_group`.

            ```
            world_size = rollout.world_size + trainer.world_size
            kwargs = {
                "rank": list(range(world_size)),
                "world_size": [world_size] * world_size,
                "master_metadata": [metadata[0]] * world_size,
            }
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def init_process_group(self, **kwargs):
        """Init process group for checkpoint engine.

        Args:
            **kwargs: Keyword arguments from `build_topology`.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """Finalize checkpoint engine after each step send_weights/receive_weights.

        1. Free weight bucket.
        1. [Optional] Deregister weight bucket for RDMA.
        2. [Optional] Destroy process group.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def bucket_size(self) -> int:
        """Return the bucket size in bytes."""
        raise NotImplementedError

    def _slice_weight_into_chunks(self, name: str, weight: torch.Tensor) -> list[tuple[torch.Tensor, dict]]:
        """Slice a large weight tensor into chunks that fit in bucket.

        Args:
            name: Name of the weight tensor.
            weight: The weight tensor to slice.

        Returns:
            List of (chunk, metadata) tuples.
        """
        from functools import reduce

        weight_size = weight.nbytes
        if weight_size <= self.bucket_size:
            # No slicing needed
            meta = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": 0,
            }
            return [(weight, meta)]

        # Slice the weight along the first dimension into chunks
        dtype_size = weight.element_size()
        numel_per_chunk = self.bucket_size // dtype_size

        # Calculate chunk size along the first dimension
        first_dim_size = weight.shape[0]
        elements_per_row = reduce(lambda x, y: x * y, weight.shape[1:], 1)
        if elements_per_row == 0:
            # Empty tensor, return as is
            meta = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": 0,
            }
            return [(weight, meta)]

        chunk_dim_size = numel_per_chunk // elements_per_row
        if chunk_dim_size == 0:
            raise ValueError(
                f"Weight '{name}' with shape {weight.shape} is too large to be chunked. A single slice "
                f"along the first dimension is larger than the bucket size ({self.bucket_size} bytes). "
                f"Please increase `checkpoint_engine.update_weights_bucket_megabytes`."
            )

        num_chunks = (first_dim_size + chunk_dim_size - 1) // chunk_dim_size
        logger.info(
            f"Slicing weight {name} ({weight.shape}, {weight.dtype}, {weight_size} bytes) into {num_chunks} chunks"
        )

        chunks = []
        start_idx = 0
        for chunk_idx in range(num_chunks):
            end_idx = min(start_idx + chunk_dim_size, first_dim_size)
            chunk = weight[start_idx:end_idx]

            meta = {
                "name": name,
                "shape": chunk.shape,
                "dtype": chunk.dtype,
                "offset": 0,  # Will be set when filling bucket
                "chunk_idx": chunk_idx,
                "total_chunks": num_chunks,
            }
            chunks.append((chunk, meta))
            start_idx = end_idx

        return chunks

    @abstractmethod
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError

    def _yield_tensors_from_buffer(
        self,
        buffer: torch.Tensor,
        bucket_meta: dict,
        pending_chunks: dict,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield tensors from buffer, handling chunk merging for large weights.

        Args:
            buffer: The buffer containing the tensor data.
            bucket_meta: The metadata of the bucket.
            pending_chunks: Dictionary to collect chunks for weights that were sliced.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        for name, meta in bucket_meta.items():
            dtype, shape = meta["dtype"], meta["shape"]
            size = dtype.itemsize * shape.numel()
            tensor = buffer[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)

            # Check if this is a chunk of a sliced weight
            if "chunk_idx" in meta and "total_chunks" in meta:
                # This is a chunk, store it for later merging
                original_name = meta["name"]
                chunk_idx = meta["chunk_idx"]
                if original_name not in pending_chunks:
                    pending_chunks[original_name] = {}
                pending_chunks[original_name][chunk_idx] = tensor

                # Check if we have all chunks for this weight
                if len(pending_chunks[original_name]) == meta["total_chunks"]:
                    # Merge all chunks back into one tensor
                    chunks_dict = pending_chunks[original_name]
                    sorted_chunks = [chunks_dict[i] for i in range(meta["total_chunks"])]
                    merged_tensor = torch.cat(sorted_chunks, dim=0)
                    yield original_name, merged_tensor
                    del pending_chunks[original_name]
            else:
                yield name, tensor


class CollectiveCheckpointEngine(CheckpointEngine):
    """Base class for collective communication checkpoint engines (NCCL, HCCL).

    This class provides common send_weights and receive_weights logic for collective
    communication backends like NCCL and HCCL.

    Subclasses must implement:
        - rank: Rank of the current process
        - send_buf: Send buffer
        - recv_buf: Receive buffer
        - _synchronize(): Synchronize device operations
        - _create_broadcast_send_op(bucket, metadata): Create broadcast operation for sending
        - _create_broadcast_recv_op(bucket): Create broadcast operation for receiving
        - _copy_to_buffer(buffer, tensor, offset): Copy tensor to buffer
    """

    @property
    @abstractmethod
    def rank(self) -> int:
        """Return the rank of the current process."""
        raise NotImplementedError

    @property
    @abstractmethod
    def send_buf(self):
        """Return the send buffer."""
        raise NotImplementedError

    @property
    @abstractmethod
    def recv_buf(self):
        """Return the receive buffer."""
        raise NotImplementedError

    @abstractmethod
    def _synchronize(self):
        """Synchronize device operations."""
        raise NotImplementedError

    @abstractmethod
    def _create_broadcast_send_op(self, bucket, metadata) -> Any:
        """Create broadcast operation for sending weights.

        Args:
            bucket: The bucket tensor to broadcast.
            metadata: The metadata to send with the bucket.

        Returns:
            A broadcast operation object with wait_for_complete method.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_broadcast_recv_op(self, bucket) -> Any:
        """Create broadcast operation for receiving weights.

        Args:
            bucket: The bucket tensor to receive data into.

        Returns:
            A broadcast operation object with wait_for_complete method that returns metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def _copy_to_buffer(self, buffer, tensor, offset):
        """Copy tensor to buffer at given offset.

        Args:
            buffer: The buffer to copy to.
            tensor: The tensor to copy.
            offset: The offset in the buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """

        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."

        # For trainer rank other than 0, consume weights without sending.
        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0

        for name, weight in weights:
            weight_size = weight.nbytes
            # Check if the weight needs to be sliced into chunks
            if weight_size > self.bucket_size:
                # Slice the weight into chunks
                chunks = self._slice_weight_into_chunks(name, weight)

                for chunk, chunk_meta in chunks:
                    chunk_size = chunk.nbytes

                    # Fill bucket with chunk
                    if offset + chunk_size > self.bucket_size:
                        self._synchronize()

                        # wait previous broadcast op finish
                        if broadcast_op is not None:
                            await broadcast_op.wait_for_complete()

                        broadcast_op = self._create_broadcast_send_op(
                            send_buf, {"bucket_meta": bucket_meta, "is_last": False}
                        )

                        # swap send_buf and recv_buf
                        send_buf, recv_buf = recv_buf, send_buf
                        bucket_meta = {}
                        offset = 0

                    # Update offset in meta (for key, we use indexed key)
                    indexed_key = f"{name}_chunk_{chunk_meta['chunk_idx']}"
                    bucket_meta[indexed_key] = {
                        "name": chunk_meta["name"],
                        "shape": chunk_meta["shape"],
                        "dtype": chunk_meta["dtype"],
                        "offset": offset,
                        "chunk_idx": chunk_meta["chunk_idx"],
                        "total_chunks": chunk_meta["total_chunks"],
                    }
                    self._copy_to_buffer(send_buf, chunk, offset)
                    offset += chunk_size

                continue

            # fill the tensor bucket
            if offset + weight_size > self.bucket_size:
                self._synchronize()

                # wait previous broadcast op finish
                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = self._create_broadcast_send_op(send_buf, {"bucket_meta": bucket_meta, "is_last": False})

                # swap send_buf and recv_buf
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            self._copy_to_buffer(send_buf, weight, offset)
            offset += weight_size

        # broadcast last bucket
        self._synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = self._create_broadcast_send_op(send_buf, {"bucket_meta": bucket_meta, "is_last": True})
        await broadcast_op.wait_for_complete()
        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        # Buffer to collect chunks for weights that were sliced
        pending_chunks = {}  # name -> {chunk_idx: tensor, ...}

        # receive first bucket
        start_time = time.time()
        broadcast_op = self._create_broadcast_recv_op(recv_buf)
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        # swap send_buf and recv_buf
        send_buf, recv_buf = recv_buf, send_buf

        while not metadata["is_last"]:
            # 1. receive next bucket
            broadcast_op = self._create_broadcast_recv_op(recv_buf)

            # 2. yield tensor from send_buf
            for tensor_tuple in self._yield_tensors_from_buffer(send_buf, metadata["bucket_meta"], pending_chunks):
                yield tensor_tuple

            # 3. wait for next bucket broadcast finish
            metadata = await broadcast_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(metadata["bucket_meta"])

            # 4. swap send_buf and recv_buf
            self._synchronize()
            send_buf, recv_buf = recv_buf, send_buf

        # yield tensor from send_buf
        for tensor_tuple in self._yield_tensors_from_buffer(send_buf, metadata["bucket_meta"], pending_chunks):
            yield tensor_tuple

        # Check if there are any remaining chunks that weren't processed
        if pending_chunks:
            raise RuntimeError(
                f"Received chunks for weights {list(pending_chunks.keys())} but did not receive all chunks for them."
            )

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )


class CheckpointEngineWithCache(CheckpointEngine):
    """Checkpoint engine with local cache: shm, disk, etc. This allow to synchronize weights without interrupting
    rollout ongoing requests (partial rollout). After requests exhausted, rollout can get weights from local cache.

    Laminar: https://arxiv.org/abs/2510.12633
    """

    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get the weights of the model from local cache.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    """Checkpoint engine for trainer and rollout colocated on same GPU.

    In trainer process:
    >>> engine = ColocatedCheckpointEngine()
    >>> trainer = Trainer()
    >>> server_adapter = ServerAdapter()
    >>> engine.send_weights(trainer.get_per_tensor_param())
    >>> server_adapter.update_weights(engine.receive_weights())
    """

    def __init__(self, bucket_size: int, is_master: bool = False) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master

    def prepare(self):
        raise NotImplementedError

    def init_process_group(self, **kwargs):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    @classmethod
    def build_topology(cls, *args, **kwargs):
        raise NotImplementedError

    def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        self.weights = weights

    def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        yield from self.weights
        self.weights = None


class CheckpointEngineWorker(Worker):
    """CheckpointEngineWorker colocated with inference engine's WorkerProc on same GPU.

    Args:
        rollout_config: The rollout configuration.
        model_config: The model configuration.
        server_adapter: The server adapter to update weights.
    """

    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        server_adapter: BaseRollout = None,
    ) -> None:
        self.rollout_config = rollout_config
        self.model_config = model_config

        # sglang and trt-llm need device_mesh for internal communication
        initialize_global_process_group_ray(timeout_second=None, backend="cpu:gloo")
        self.server_adapter: BaseRollout = server_adapter or get_rollout_class(
            rollout_config.name, rollout_config.mode
        )(config=rollout_config, model_config=model_config, device_mesh=None)

        backend = rollout_config.checkpoint_engine.backend
        bucket_size = rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
        engine_kwargs = rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})
        self.checkpoint_engine = CheckpointEngineRegistry.new(backend, bucket_size=bucket_size, **engine_kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        weights = self.checkpoint_engine.receive_weights()
        await self.server_adapter.update_weights(weights)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)


_worker_cls = ray.remote(CheckpointEngineWorker)


class CheckpointEngineManager:
    """Checkpoint engine manager to coordinate weight synchronization between trainer and rollout replicas.

    - ME: model engine, FSDP, MCore, VeOmni, export full tensor generator `get_per_tensor_param`
    - CE: checkpoint engine, NCCL, NIXL, etc

    In trainer, model engine and checkpoint engine are in same process.
    In rollout, checkpoint engine and rollout worker are in separate process, update weights via cuda ipc.

    ```
    ┌────────┬────────┬─────┬────────┐         ┌───────────────────┬───────────────────┐
    │ ┌────┐ │ ┌────┐ │     │ ┌────┐ │         │     Replica 0     │     Replica 1     │
    │ │ ME0│ │ │ ME1│ │     │ │ MEn│ │         ├────┬────┬────┬────┼────┬────┬────┬────┤
    │ └──┬─┘ │ └────┘ │ ... │ └────┘ │         │ 0  │ 1  │ 2  │ 3  │ 0  │ 1  │ 2  │ 3  │
    │    v   |        |     |        |         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
    | ┌──┴─┐ │ ┌────┐ │     │ ┌────┐ │            ^    ^    ^   cuda ipc   ^    ^    ^
    │ │ CE │ │ │ CE │ │     │ │ CE │ │         ┌──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┐
    │ └──┬─┘ │ └────┘ │     │ └────┘ │         │ CE │ CE │ CE │ CE │ CE │ CE │ CE │ CE |
    └────┼───┴────────┴─────┴────────┘         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
         v                                        |    |    |    |    |    |    |    |
         └─────────────(nccl/nixl/..)─────────────┴────┴────┴────┴────┴────┴────┴────┘
    ```

    Args:
        backend: The checkpoint engine backend.
        trainer: The trainer worker group.
        replicas: The list of rollout replicas.
    """

    def __init__(
        self,
        backend: str,
        trainer: RayWorkerGroup,
        replicas: list[RolloutReplica],
    ) -> None:
        self.backend = backend
        self.backend_cls = CheckpointEngineRegistry.get(backend)
        self.trainer = trainer
        self.replicas = replicas

    def build_process_group(self, rollout: RayWorkerGroup):
        """Build process group for trainer and rollout replicas."""
        trainer = self.trainer

        # 1. prepare all workers
        metadata = ray.get(
            trainer.execute_checkpoint_engine(["prepare"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        )

        # 2. build communication topology between all workers
        trainer_kwargs, rollout_kwargs = self.backend_cls.build_topology(
            trainer.world_size, rollout.world_size, metadata
        )
        for k, v in trainer_kwargs.items():
            assert len(v) == trainer.world_size, f"trainer_kwargs[{k}] must have length of {trainer.world_size}"
        for k, v in rollout_kwargs.items():
            assert len(v) == rollout.world_size, f"rollout_kwargs[{k}] must have length of {rollout.world_size}"

        trainer_kwargs["method"] = ["init_process_group"] * trainer.world_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size

        # 3. init process group between all workers
        ray.get(
            trainer.execute_checkpoint_engine(**trainer_kwargs) + rollout.execute_checkpoint_engine(**rollout_kwargs)
        )

    def add_replicas(self, replicas: list[RolloutReplica]):
        """Add rollout replicas to the manager for elastic scale up, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to add.
        """
        self.replicas.extend(replicas)

    def remove_replicas(self, replicas: list[RolloutReplica]):
        """Remove rollout replicas from the manager for elastic scale down, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to remove.
        """
        replicas_set = set(replicas)
        self.replicas = [r for r in self.replicas if r not in replicas_set]

    @auto_await
    async def sleep_replicas(self):
        """Sleep all rollout replicas: free weight and kv_cache device memory."""
        # skip sleep replicas for disaggregated rollout
        if self.backend != "naive":
            return
        await asyncio.gather(*[r.sleep() for r in self.replicas])

    @auto_await
    async def update_weights(self):
        """Update weights from trainer to rollout replicas."""

        # 0. update weights for sync training with colocated trainer and rollout
        if self.backend == "naive":
            ray.get(self.trainer.update_weights())
            return

        # 1. abort and save all unfinished requests for partial rollout
        await asyncio.gather(*[r.abort_all_requests() for r in self.replicas])

        # 2. create a temporay worker group for all replicas
        workers = []
        for replica in self.replicas:
            workers.extend(replica.workers)
        rollout = RayWorkerGroup(worker_handles=workers, ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls))
        trainer = self.trainer

        # 3. build process group
        self.build_process_group(rollout)

        # 4. update weights of all workers
        ray.get(trainer.update_weights() + rollout.update_weights())

        # 5. finalize all workers
        ray.get(
            trainer.execute_checkpoint_engine(["finalize"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
        )

        # 6. resume all unfinished requests for partial rollout
        await asyncio.gather(*[r.resume_all_requests() for r in self.replicas])
