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
from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import pickle
import threading
from contextlib import asynccontextmanager
from typing import Any, Generator, Optional

import aiohttp
import pynvml
import ray
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default configuration constants
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_MAX_CONNECTIONS = 2000
DEFAULT_MAX_WAIT_TIME = 300.0


@contextlib.contextmanager
def nvml_context():
    """Context manager for NVML initialization and shutdown.

    Raises:
        RuntimeError: If NVML initialization fails
    """
    try:
        pynvml.nvmlInit()
        yield
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to initialize NVML: {e}") from e
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


_NVML_INITIALIZED = False
_NVML_LOCK = threading.Lock()


def get_device_uuid(id: int) -> str:
    """Get the UUID of a CUDA device using NVML."""
    global _NVML_INITIALIZED
    with _NVML_LOCK:
        if not _NVML_INITIALIZED:
            try:
                pynvml.nvmlInit()
                _NVML_INITIALIZED = True
            except pynvml.NVMLError as e:
                raise RuntimeError(f"Failed to initialize NVML: {e}") from e

    # Get the device handle and UUID
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        # Ensure the UUID is returned as a string, not bytes
        if isinstance(uuid, bytes):
            return uuid.decode("utf-8")
        elif isinstance(uuid, str):
            return uuid
        else:
            raise RuntimeError(f"Unexpected UUID type: {type(uuid)} for device {id} (global index: {id})")
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to get device UUID for device {id} (global index: {id}): {e}") from e


async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    if resp.status == 204 or (resp.content_length == 0):
        return {}

    try:
        return await resp.json(content_type=None)
    except Exception:
        try:
            text = await resp.text()
        except Exception:
            return {}
        return {
            "content_type": (resp.headers.get("Content-Type") or ""),
            "text": text,
        }


class AsyncTRTLLMHttpAdapter:
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.max_connections = max_connections

    @asynccontextmanager
    async def _get_session(self) -> aiohttp.ClientSession:
        """Context manager for safe session access with proper connection pooling.

        Yields:
            aiohttp.ClientSession: Session instance for making HTTP requests

        Note:
            This method creates a new session for each request to avoid resource competition
            while still maintaining proper connection pooling through the shared connector.
        """
        # Create a new session for each request to avoid resource competition
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections // 4,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        try:
            yield session
        finally:
            # Always close the session to free up resources
            if not session.closed:
                await session.close()

    async def _make_async_request(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        method: str = "POST",
        return_status: bool = False,
    ) -> dict[str, Any] | int:
        """Make an async HTTP request with retry logic and consistent error handling.

        Args:
            endpoint (str): The API endpoint to call (without leading slash)
            payload (Optional[Dict[str, Any]], optional): The JSON payload to send.
                Defaults to empty dict if None.
            method (str, optional): HTTP method to use. Defaults to "POST".

        Returns:
            Dict[str, Any]: The JSON response from the server

        Raises:
            aiohttp.ClientResponseError: If the HTTP request fails with a client/server error
            RuntimeError: If all retry attempts are exhausted

        Note:
            - Uses exponential backoff for retries
            - Logs warnings for timeout and connection errors, errors for HTTP errors
        """

        url = f"http://{self.host}:{self.port}/{endpoint}"

        for attempt in range(self.max_attempts):
            try:
                async with self._get_session() as session:
                    if method.upper() == "GET":
                        async with session.get(url, timeout=timeout) as response:
                            response.raise_for_status()
                            return response.status if return_status else await _read_async_response(response)
                    else:
                        async with session.post(url, json=payload or {}, timeout=timeout) as response:
                            response.raise_for_status()
                            return response.status if return_status else await _read_async_response(response)

            except asyncio.TimeoutError:
                logger.warning(f"Async request to {endpoint} timed out (attempt {attempt + 1})")
            except aiohttp.ClientConnectorError:
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_attempts - 1:
                    raise

            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(f"Failed to complete async request to {endpoint} after {self.max_attempts} attempts")

    async def resume_memory_occupation(self, tags: list[str]):
        """Resume GPU memory occupation (async version).

        Similar to AsyncEngine, this method handles first-time weight reloading
        by calling release_memory_occupation if needed.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to resume.
                If None, resumes all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory resume status
        """
        return await self._make_async_request("resume_memory", {"tags": tags})

    async def release_memory_occupation(self, tags: list[str]):
        """Release GPU memory occupation temporarily (async version).

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to release.
                If None, releases all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory release status
        """
        return await self._make_async_request("release_memory", {"tags": tags})

    async def update_weights(self, weights: dict[str, str]):
        """Update model weights from tensor data asynchronously.

        Args:
            weights: A dictionary that maps the device uuid of the weight handles.

        Returns:
            Dict[str, Any]: Server response containing update status
        """
        return await self._make_async_request("update_weights", {"weights": weights})


class ServerAdapter(BaseRollout):
    _WEIGHTS_TAGS = [
        "sampler",
        "drafter",
        "guided_decoder",
        "spec_resource_manager",
        "model_extra",
        "executor_extra",
        "model",
        "draft_model",
    ]

    @staticmethod
    def get_full_tags() -> list[str]:
        return ServerAdapter._WEIGHTS_TAGS + ["kv_cache"]

    def __init__(
        self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh, replica_rank: int = -1
    ):
        super().__init__(config, model_config, device_mesh)
        self._adapter = None
        self.hybrid_device_mesh = None
        self.gpu_id = None
        self.is_leader_rank = None
        self.replica_rank = None
        self.is_dp_rank = None

        # hybrid mode
        if self.device_mesh is not None:
            assert device_mesh.mesh_dim_names.index("dp") == 0, "DP dim should always be the first dimension"

            # Clone a new device mesh for CPU backend only (used for internal ranks communication)
            device_mesh_kwargs = dict(
                mesh_shape=device_mesh.mesh.shape,
                mesh_dim_names=device_mesh.mesh_dim_names,
            )
            self.hybrid_device_mesh = init_device_mesh("cpu", **device_mesh_kwargs)

            self.hybrid_device_mesh[self.hybrid_device_mesh.mesh_dim_names[1:]]._flatten(mesh_dim_name="exclude_dp")
            self.is_leader_rank = self.hybrid_device_mesh["exclude_dp"].get_local_rank() == 0
            logger.info(f"is_dp_leader: {self.is_leader_rank}")
            logger.info(f"exclude_dp_rank = {self.hybrid_device_mesh['exclude_dp'].get_local_rank()}")
            logger.info(f"exclude_dp_size = {self.hybrid_device_mesh['exclude_dp'].size()}")
            self.gpu_id = ray.get_gpu_ids()[0]
            self.replica_rank = self.hybrid_device_mesh["dp"].get_local_rank()
            assert len(ray.get_gpu_ids()) == 1, "ServerAdapter should run on a single GPU node"
        else:
            rank = int(os.environ["RANK"])
            self.replica_rank = replica_rank
            self.is_leader_rank = rank == 0

        # Below is required for all modes.
        assert self.replica_rank >= 0, "replica_rank is not set"
        assert self.is_leader_rank is not None, "is_leader_rank is not set"

        self.node_ip = ray.util.get_node_ip_address().strip("[]")

    async def _init_server_adapter(self):
        if self._adapter is not None:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"trtllm_server_{self.replica_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        assert server_address == self.node_ip, f"server address: {server_address} != node_ip: {self.node_ip}"

        logger.debug(f"replica_rank={self.replica_rank}, server address: {server_address}, port: {server_port}")
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._adapter = AsyncTRTLLMHttpAdapter(
            host=host,
            port=server_port,
        )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        if self.is_leader_rank and self.config.free_cache_engine:
            if "weights" in tags:
                tags = self._WEIGHTS_TAGS
            elif "kv_cache" in tags:
                tags = ["kv_cache"]
            else:
                raise ValueError(f"Invalid tag: {tags}")
            await self._init_server_adapter()
            await self._adapter.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.is_leader_rank and self.config.free_cache_engine:
            await self._init_server_adapter()
            tags = self._WEIGHTS_TAGS + ["kv_cache"]
            await self._adapter.release_memory_occupation(tags=tags)

    async def update_weights_from_ipc_handles(self, device_handles):
        assert self.hybrid_device_mesh is not None, "hybrid_device_mesh is not set"

        """Update weights from IPC handles."""
        if self.is_leader_rank:
            gathered_handles = [None for _ in range(self.hybrid_device_mesh["exclude_dp"].size())]
        else:
            gathered_handles = None

        await asyncio.to_thread(
            dist.gather_object,
            obj=device_handles,
            object_gather_list=gathered_handles,
            group_dst=0,
            group=self.hybrid_device_mesh["exclude_dp"].get_group(),
        )

        if self.is_leader_rank:
            all_handles = {k: v for d in gathered_handles for k, v in d.items()}
            await self._adapter.update_weights(all_handles)

        await asyncio.to_thread(dist.barrier, group=self.hybrid_device_mesh["exclude_dp"].get_group())

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        assert self.hybrid_device_mesh is not None, "hybrid_device_mesh is not set"

        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        if self.is_leader_rank:
            await self._init_server_adapter()

        total_available_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) * 1024 * 1024

        try:
            device_uuid = get_device_uuid(self.gpu_id)
        except Exception as e:
            logger.error(f"Failed to get device UUID in update_weights(): {e}")
            device_uuid = None
            raise e

        cur_available_bytes = total_available_bytes
        cur_handles = []

        async def flush():
            nonlocal cur_available_bytes, cur_handles
            if not cur_handles:
                return
            serialized_device_handles = {device_uuid: base64.b64encode(pickle.dumps(cur_handles)).decode("utf-8")}
            await self.update_weights_from_ipc_handles(serialized_device_handles)
            cur_available_bytes = total_available_bytes
            cur_handles = []

        for name, param in weights:
            size_in_bytes = param.element_size() * param.numel()
            if size_in_bytes > cur_available_bytes:
                await flush()

            assert cur_available_bytes >= size_in_bytes, (
                f"cur_available_bytes: {cur_available_bytes:,} size_in_bytes: {size_in_bytes:,} name: {name}"
            )
            cur_available_bytes -= size_in_bytes
            handle = reduce_tensor(param.detach())
            cur_handles.append((name, handle))

        await flush()

        if self.is_leader_rank:
            # Finalize update weights
            await self._adapter.update_weights(None)
        await asyncio.to_thread(dist.barrier, group=self.hybrid_device_mesh["exclude_dp"].get_group())
