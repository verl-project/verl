# Copyright 2025 z.ai
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
#
# This file is adapted from multiple sources:
# 1. THUDM/slime project
#    Original source: https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/http_server_engine.py
#    Copyright 2025 z.ai
#    Licensed under the Apache License, Version 2.0
# 2. SGLang project
#    Original source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server_engine.py
#    Copyright 2023-2024 SGLang Team
#    Licensed under the Apache License, Version 2.0
#
# Modifications made by z.ai and ModelBest Inc. include but are not limited to:
# - Enhanced error handling and retry logic
# - Added async support with connection pooling
# - Extended functionality for distributed weight updates
# - Improved logging and monitoring capabilities
# - Additional configuration options and optimizations

"""HTTP Server Engine Adapter for SGLang.

This module provides HTTP-based adapters for SGLang engines, allowing communication
with SGLang servers through HTTP requests instead of direct engine calls.

Classes:
    HttpServerAdapter: Synchronous HTTP adapter for SGLang engines
    AsyncHttpServerAdapter: Asynchronous HTTP adapter for SGLang engines

Functions:
    launch_server_process: Launch and initialize an SGLang HTTP server process
"""

import asyncio
import logging
import multiprocessing
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

import aiohttp
import requests
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput, UpdateWeightsFromTensorReqInput
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default configuration constants
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_MAX_CONNECTIONS = 2000
DEFAULT_MAX_WAIT_TIME = 300.0


def _read_response(response: requests.Response):
    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError:
        return {
            "content_type": response.headers.get("Content-Type", ""),
            "text": response.text,
        }


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


def launch_server_process(
    server_args: ServerArgs,
    timeout: float = DEFAULT_TIMEOUT,
    max_wait_time=DEFAULT_MAX_WAIT_TIME,
    first_rank_in_node=False,
) -> multiprocessing.Process:
    """Launch an SGLang HTTP server process and wait for it to be ready.

    This function starts a new process running an SGLang HTTP server, then waits
    for the server to become ready by polling its health endpoints. It ensures
    the server is fully operational before returning.

    Args:
        server_args (ServerArgs): Server configuration arguments including host, port, and other settings
        timeout (float, optional): Timeout for individual HTTP requests during health checks.
            Defaults to DEFAULT_TIMEOUT.

    Returns:
        multiprocessing.Process: The launched multiprocessing.Process instance

    Raises:
        RuntimeError: If the server process terminates unexpectedly during startup or cache flush
        TimeoutError: If server fails to become ready within reasonable time (300 seconds)
        requests.RequestException: If health check requests fail repeatedly

    Note:
        This function will return immediately for non-master nodes (node_rank != 0),
        but the process will still be started and returned.
        This is for consistency; except for the process obtained by node_rank = 0,
        other processes have no actual effect.
    """
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    if server_args.node_rank != 0 or not first_rank_in_node:
        logger.info(f"Server process started with PID {p.pid} for node rank {server_args.node_rank}", flush=True)
        return p

    p.start()

    base_url = server_args.url()
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {server_args.api_key}",
    }

    # Health check with overall timeout
    start_time = time.time()

    with requests.Session() as session:
        while time.time() - start_time < max_wait_time:
            if not p.is_alive():
                raise RuntimeError("Server process terminated unexpectedly during startup")

            try:
                if server_args.is_embedding:
                    response = session.get(f"{base_url}/health", headers=headers, timeout=timeout)
                else:
                    response = session.get(f"{base_url}/health_generate", headers=headers, timeout=timeout)
                if response.status_code == 200:
                    break
            except requests.RequestException as e:
                logger.debug(f"Health check failed: {e}")

            time.sleep(2)
        else:
            p.terminate()
            logger.error(f"Server in {base_url} failed to become healthy within timeout period")
            raise TimeoutError("Server failed to become healthy within timeout period")

        # Ensure cache is ready
        while time.time() - start_time < max_wait_time:
            if not p.is_alive():
                raise RuntimeError("Server process terminated unexpectedly during cache flush")

            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers, timeout=timeout)
                if response.status_code == 200:
                    break
            except requests.RequestException as e:
                logger.debug(f"Cache flush check failed: {e}")

            time.sleep(2)
        else:
            p.terminate()
            raise TimeoutError("Server cache flush failed within timeout period")

    return p


class HttpServerAdapter(EngineBase):
    """HTTP-based adapter for SGLang engines.

    This adapter allows interaction with SGLang engines through HTTP requests
    instead of direct engine calls. It launches an HTTP server process and
    provides methods to communicate with it via REST API calls.

    You can use this class to launch a server from a HttpServerAdapter instance.
    We recommend using this class only when you need to use http server.
    Otherwise, you can use Engine directly.

    Attributes:
        router_ip (Optional[str]): IP address of the router for worker registration
        router_port (Optional[int]): Port of the router for worker registration
        server_args (ServerArgs): Server configuration arguments
        node_rank (int): Rank of this node in distributed setup
        process (multiprocessing.Process): The launched server process
        timeout (float): HTTP request timeout in seconds
        max_attempts (int): Maximum number of attempts for requests
        retry_delay (float): Base delay between retries in seconds
    """

    def __init__(
        self,
        router_ip: Optional[str] = None,
        router_port: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        first_rank_in_node: bool = False,
        max_start_wait_time: float = DEFAULT_MAX_WAIT_TIME,
        launch_server: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the HTTP server engine adapter.

        Args:
            router_ip (Optional[str], optional): IP address of router for worker registration.
                Defaults to None.
            router_port (Optional[int], optional): Port of router for worker registration.
                Defaults to None.
            timeout (float, optional): HTTP request timeout in seconds.
                Defaults to DEFAULT_TIMEOUT.
            max_attempts (int, optional): Maximum number of retry attempts for failed requests.
                Defaults to DEFAULT_MAX_ATTEMPTS.
            retry_delay (float, optional): Base delay between retries in seconds.
                Defaults to DEFAULT_RETRY_DELAY.
            launch_server (bool, optional): Whether to launch the server process.
                Defaults to True.
            **kwargs (Any): Additional arguments passed to ServerArgs

        Note:
            TODO: @ChangyiYang Enable SGLang router for this http server engine
            If both router_ip and router_port are provided and this is the master node
            (node_rank == 0), the adapter will automatically register with the router.
        """
        self.router_ip: Optional[str] = router_ip
        self.router_port: Optional[int] = router_port
        self.timeout: float = timeout
        self.max_attempts: int = max_attempts
        self.retry_delay: float = retry_delay
        self.server_args: ServerArgs = ServerArgs(**kwargs)
        self.node_rank: int = self.server_args.node_rank
        self.max_start_wait_time: float = max_start_wait_time

        logger.info(
            f"Launch HttpServerAdapter at: {self.server_args.host}:{self.server_args.port} with {first_rank_in_node}"
        )
        if launch_server:
            self.process: multiprocessing.Process = launch_server_process(
                self.server_args, self.timeout, self.max_start_wait_time, first_rank_in_node
            )

        if self.node_rank == 0 and self.router_ip and self.router_port:
            self._register_with_router()

    def _register_with_router(self) -> None:
        """Register worker with router with error handling.

        This method attempts to register the current worker with a router service.
        If registration fails, it logs an error but does not raise an exception,
        allowing the server to continue operating without router integration.

        Raises:
            Does not raise exceptions - all errors are logged and handled gracefully.
        """
        try:
            url = f"http://{self.router_ip}:{self.router_port}/add_worker"
            params = {"url": f"http://{self.server_args.host}:{self.server_args.port}"}
            response = requests.post(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            logger.info("Successfully registered with router")
        except Exception as e:
            logger.error(f"Failed to register with router: {e}")
            # Don't raise here - server can still work without router

    def _make_request(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
        method: str = "POST",
        timeout: float = DEFAULT_TIMEOUT,
        only_master: bool = True,
    ) -> dict[str, Any]:
        """Make a HTTP request with retry logic and consistent error handling.

        Args:
            endpoint (str): The API endpoint to call (without leading slash)
            payload (Optional[Dict[str, Any]], optional): The JSON payload to send.
                Defaults to empty dict if None.
            method (str, optional): HTTP method to use. Defaults to "POST".

        Returns:
            Dict[str, Any]: The JSON response from the server

        Raises:
            requests.HTTPError: If the HTTP request fails with a client/server error
            RuntimeError: If all retry attempts are exhausted

        Note:
            - For non-master nodes (node_rank != 0), returns empty dict immediately
            - Uses exponential backoff for retries
            - Logs warnings for timeout and connection errors, errors for HTTP errors
        """
        if only_master and self.node_rank != 0:
            return {}

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

        for attempt in range(self.max_attempts):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, timeout=self.timeout)
                else:
                    response = requests.post(url, json=payload or {}, timeout=self.timeout)

                response.raise_for_status()
                return _read_response(response)

            except requests.exceptions.Timeout:
                logger.warning(f"Request to {endpoint} timed out (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_attempts - 1:
                    raise

            if attempt < self.max_attempts - 1:
                time.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(f"Failed to complete request to {endpoint} after {self.max_attempts} attempts")

    def update_weights_from_tensor(self, req: UpdateWeightsFromTensorReqInput) -> dict[str, Any]:
        """Update model weights from tensor data.

        The HTTP server will only post meta data, and the real weights will be
        copied directly from GPUs.

        Args:
            serialized_named_tensors (List[str]): List of serialized tensor data
            load_format (Optional[str], optional): Format specification for loading weights.
                Defaults to None.
            flush_cache (bool, optional): Whether to flush cache after updating weights.
                Defaults to False.

        Returns:
            Dict[str, Any]: Server response containing update status

        Note:
            The model should be on GPUs rather than CPU for this functionality to work properly.
            If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        import base64

        named_tensors = req.serialized_named_tensors
        load_format = req.load_format
        flush_cache = req.flush_cache

        if named_tensors:
            serialized_named_tensors = [
                base64.b64encode(named_tensor).decode("utf-8") for named_tensor in named_tensors
            ]
        else:
            serialized_named_tensors = []

        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def shutdown(self) -> None:
        """Shutdown the HTTP server and clean up resources.

        This method performs the following cleanup operations:
        1. Unregisters the worker from the router (if configured)
        2. Terminates the server process tree

        All operations are performed with error handling to ensure graceful shutdown
        even if individual steps fail.

        Note:
            This method should be called when the adapter is no longer needed
            to ensure proper cleanup of resources and processes.
        """
        # Unregister from router
        if self.router_ip and self.router_port:
            try:
                url = f"http://{self.router_ip}:{self.router_port}/remove_worker"
                params = {"url": f"http://{self.server_args.host}:{self.server_args.port}"}
                requests.post(url, params=params, timeout=5.0)  # Short timeout for shutdown
                logger.info("Successfully unregistered from router")
            except Exception as e:
                logger.warning(f"Failed to unregister from router: {e}")

        # Kill server process
        if hasattr(self, "process") and self.process is not None:
            try:
                kill_process_tree(self.process.pid)
                logger.info("Server process terminated")
            except Exception as e:
                logger.error(f"Failed to terminate server process: {e}")

    def generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[dict[str, Any]] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        token_ids_logprob: Optional[list[int]] = None,
        lora_path: Optional[str] = None,
        custom_logit_processor: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """Generate text using the SGLang server.

        Args:
            prompt (Optional[str], optional): Text prompt for generation. Defaults to None.
            sampling_params (Optional[Dict[str, Any]], optional): Parameters controlling
                text generation sampling. Defaults to None.
            input_ids (Optional[List[int]], optional): Alternative to prompt, direct token IDs input.
                Defaults to None.
            image_data (Optional[Any], optional): Image data for multimodal generation.
                Defaults to None.
            return_logprob (bool, optional): Whether to return log probabilities.
                Defaults to False.
            logprob_start_len (Optional[int], optional): Starting length for log probability calculation.
                Defaults to None.
            top_logprobs_num (Optional[int], optional): Number of top log probabilities to return.
                Defaults to None.
            token_ids_logprob (Optional[List[int]], optional): Specific token IDs for
                log probability calculation. Defaults to None.
            lora_path (Optional[str], optional): Path to LoRA adapter weights. Defaults to None.
            custom_logit_processor (Optional[Callable], optional): Custom logit processing function.
                Defaults to None.

        Returns:
            Dict[str, Any]: Generated text and associated metadata from the server

        Note:
            Either prompt or input_ids should be provided, but not both.
            The response format depends on the server configuration and parameters.
        """
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._make_request("generate", payload, only_master=False)

    def reward_score(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        lora_path: Optional[str] = None,
    ) -> dict[str, Any]:
        assert self.server_args.is_embedding, "Score is only supported for embedding models"
        payload = {
            "text": prompt,
            "input_ids": input_ids,
            "image_data": image_data,
            "lora_path": lora_path,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._make_request("classify", payload, only_master=False)

    def flush_cache(self) -> dict[str, Any]:
        """Flush the cache of the server.

        This method repeatedly attempts to flush the server cache until successful.
        The flush operation will not return status 200 when there are pending requests.

        Returns:
            Dict[str, Any]: Server response indicating cache flush status.
                For non-master nodes, returns empty dict.

        Note:
            Uses retry logic with limited attempts (max_attempts * 2) to avoid infinite loops.
            Each retry includes a delay to allow pending requests to complete.
        """
        if self.node_rank != 0:
            return {}

        # Use retry logic with limited attempts to avoid infinite loops
        for attempt in range(self.max_attempts * 2):  # Allow more retries for cache flush
            try:
                response = requests.get(
                    f"http://{self.server_args.host}:{self.server_args.port}/flush_cache", timeout=self.timeout
                )
                if response.status_code == 200:
                    return _read_response(response)
            except Exception as e:
                logger.warning(f"Error flushing cache (attempt {attempt + 1}): {e}")

            time.sleep(self.retry_delay)

        logger.error("Failed to flush cache after maximum attempts")
        return {}

    def release_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        """Release GPU memory occupation temporarily.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to release.
                If None, releases all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory release status
        """
        return self._make_request("release_memory_occupation", {"tags": tags})

    def resume_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        """Resume GPU memory occupation.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to resume.
                If None, resumes all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory resume status
        """
        return self._make_request("resume_memory_occupation", {"tags": tags})

    def abort_request(self, rid: str = "", abort_all: bool = False) -> dict[str, Any]:
        """Abort a request.

        Args:
            rid (str): The ID of the request to abort
            abort_all (bool, optional): Whether to abort all requests. Defaults to False.

        Returns:
            Dict[str, Any]: Server response indicating abort status
        """
        return self._make_request("abort_request", {"rid": rid, "abort_all": abort_all})


class AsyncHttpServerAdapter(HttpServerAdapter):
    """Asynchronous HTTP-based adapter for SGLang engines.

    This class inherits from HttpServerAdapter and adds async capabilities
    for non-blocking HTTP requests to the SGLang server. It provides the same
    functionality as the synchronous version but with async/await support.

    The async adapter is useful when you need to make multiple concurrent requests
    or integrate with async frameworks. It uses aiohttp for efficient async HTTP
    communication and maintains connection pooling for better performance.

    Attributes:
        max_connections (int): Maximum number of connections in the connection pool
    """

    def __init__(
        self,
        router_ip: Optional[str] = None,
        router_port: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        first_rank_in_node: bool = False,
        launch_server: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the async HTTP server engine adapter.

        Args:
            router_ip (Optional[str], optional): IP address of router for worker registration.
                Defaults to None.
            router_port (Optional[int], optional): Port of router for worker registration.
                Defaults to None.
            timeout (float, optional): HTTP request timeout in seconds.
                Defaults to DEFAULT_TIMEOUT.
            max_attempts (int, optional): Maximum number of retry attempts for failed requests.
                Defaults to DEFAULT_MAX_ATTEMPTS.
            retry_delay (float, optional): Base delay between retries in seconds.
                Defaults to DEFAULT_RETRY_DELAY.
            max_connections (int, optional): Maximum number of connections in the connection pool.
                Defaults to DEFAULT_MAX_CONNECTIONS.
            launch_server (bool, optional): Whether to launch the server process.
                Defaults to True.
            **kwargs (Any): Additional arguments passed to ServerArgs
        """
        super().__init__(
            router_ip,
            router_port,
            timeout,
            max_attempts,
            retry_delay,
            first_rank_in_node,
            launch_server=launch_server,
            **kwargs,
        )
        self.max_connections: int = max_connections

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
        method: str = "POST",
        timeout: float = DEFAULT_TIMEOUT,
        only_master: bool = True,
    ) -> dict[str, Any]:
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
            - For non-master nodes (node_rank != 0), returns empty dict immediately
            - Uses exponential backoff for retries
            - Logs warnings for timeout and connection errors, errors for HTTP errors
        """
        if only_master and self.node_rank != 0:
            return {}

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

        for attempt in range(self.max_attempts):
            try:
                async with self._get_session() as session:
                    if method.upper() == "GET":
                        async with session.get(url, timeout=timeout) as response:
                            response.raise_for_status()
                            return await _read_async_response(response)
                    else:
                        async with session.post(url, json=payload or {}, timeout=timeout) as response:
                            response.raise_for_status()
                            return await _read_async_response(response)

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

    async def release_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        """Release GPU memory occupation temporarily (async version).

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to release.
                If None, releases all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory release status
        """
        return await self._make_async_request("release_memory_occupation", {"tags": tags})

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        """Resume GPU memory occupation (async version).

        Similar to AsyncEngine, this method handles first-time weight reloading
        by calling release_memory_occupation if needed.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to resume.
                If None, resumes all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory resume status
        """
        return await self._make_async_request("resume_memory_occupation", {"tags": tags})

    async def update_weights_from_tensor(
        self,
        req: UpdateWeightsFromTensorReqInput,
    ) -> dict[str, Any]:
        """Update model weights from tensor data asynchronously.

        Args:
            serialized_named_tensors (List[str]): List of serialized tensor data
            load_format (Optional[str], optional): Format specification for loading weights.
                Defaults to None.
            flush_cache (bool, optional): Whether to flush cache after updating weights.
                Defaults to True.

        Returns:
            Dict[str, Any]: Server response containing update status
        """
        import base64

        named_tensors = req.serialized_named_tensors
        load_format = req.load_format
        flush_cache = req.flush_cache

        serialized_named_tensors = [base64.b64encode(named_tensor).decode("utf-8") for named_tensor in named_tensors]
        return await self._make_async_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    async def load_lora_adapter_from_tensor(self, req: LoadLoRAAdapterFromTensorsReqInput):
        return await self._make_async_request(
            "load_lora_adapter_from_tensors",
            {
                "lora_name": req.lora_name,
                "config_dict": req.config_dict,
                "serialized_tensors": req.serialized_tensors,
            },
        )

    async def unload_lora_adapter(self, lora_name: str):
        return await self._make_async_request(
            "unload_lora_adapter",
            {
                "lora_name": lora_name,
            },
        )

    async def available_models(self):
        return await self._make_async_request(endpoint="v1/models", method="GET")

    async def flush_cache(self) -> dict[str, Any]:
        """Flush the cache of the server asynchronously.

        Similar to the sync version, this method retries until the cache
        is successfully flushed. It uses async sleep between retries.

        Returns:
            Dict[str, Any]: Server response indicating cache flush status.
                For non-master nodes, returns empty dict.

        Note:
            Uses retry logic with limited attempts (max_attempts * 4) to avoid infinite loops.
            Each retry includes an async delay to allow pending requests to complete.
        """
        if self.node_rank != 0:
            return {}

        # Use retry logic with limited attempts to avoid infinite loops
        for attempt in range(self.max_attempts * 4):  # Allow more retries for cache flush
            try:
                async with self._get_session() as session:
                    url = f"http://{self.server_args.host}:{self.server_args.port}/flush_cache"
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await _read_async_response(response)
            except Exception as e:
                logger.warning(f"Error flushing cache (attempt {attempt + 1}): {e}")

            await asyncio.sleep(self.retry_delay)

        logger.error("Failed to flush cache after maximum attempts")
        return {}

    async def generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[dict[str, Any]] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        token_ids_logprob: Optional[list[int]] = None,
        lora_path: Optional[str] = None,
        custom_logit_processor: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """Generate text using the SGLang server asynchronously."""
        logger.info("generate() started")

        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }

        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Send request
        response = await self._make_async_request("generate", payload, timeout=self.timeout, only_master=False)

        return response

    async def async_generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[dict[str, Any]] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        token_ids_logprob: Optional[list[int]] = None,
        lora_path: Optional[str] = None,
        custom_logit_processor: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """Async generate method that mirrors AsyncEngine.async_generate interface.

        This method provides compatibility with AsyncEngine's async_generate method
        by forwarding the call to the generate method. It ensures API consistency
        between direct engine usage and HTTP-based engine usage.

        Args:
            prompt (Optional[str], optional): Text prompt for generation. Defaults to None.
            sampling_params (Optional[Dict[str, Any]], optional): Parameters controlling
                text generation sampling. Defaults to None.
            input_ids (Optional[List[int]], optional): Alternative to prompt, direct token IDs input.
                Defaults to None.
            image_data (Optional[Any], optional): Image data for multimodal generation.
                Defaults to None.
            return_logprob (bool, optional): Whether to return log probabilities.
                Defaults to False.
            logprob_start_len (Optional[int], optional): Starting length for log probability calculation.
                Defaults to None.
            top_logprobs_num (Optional[int], optional): Number of top log probabilities to return.
                Defaults to None.
            token_ids_logprob (Optional[List[int]], optional): Specific token IDs for
                log probability calculation. Defaults to None.
            lora_path (Optional[str], optional): Path to LoRA adapter weights. Defaults to None.
            custom_logit_processor (Optional[Callable], optional): Custom logit processing function.
                Defaults to None.

        Returns:
            Dict[str, Any]: Generated text and associated metadata from the server

        Note:
            This method is provided for API compatibility with AsyncEngine.
            It forwards all calls to the generate method.
        """
        return await self.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            input_ids=input_ids,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
        )

    async def reward_score(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        lora_path: Optional[str] = None,
    ) -> dict[str, Any]:
        logger.info("reward_score() started")
        payload = {
            "text": prompt,
            "input_ids": input_ids,
            "image_data": image_data,
            "lora_path": lora_path,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Send request
        response = await self._make_async_request("classify", payload, timeout=self.timeout, only_master=False)

        return response

    async def async_reward_score(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        lora_path: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self.reward_score(
            prompt=prompt,
            input_ids=input_ids,
            image_data=image_data,
            lora_path=lora_path,
        )

    async def abort_request(self, rid: str = "", abort_all: bool = False) -> dict[str, Any]:
        """Abort a request asynchronously.

        Args:
            rid (str): The ID of the request to abort
            abort_all (bool, optional): Whether to abort all requests. Defaults to False.

        Returns:
            Dict[str, Any]: Server response indicating abort status
        """
        return await self._make_async_request("abort_request", {"rid": rid, "abort_all": abort_all})
