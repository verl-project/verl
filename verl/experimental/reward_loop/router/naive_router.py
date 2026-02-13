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
import multiprocessing
import os
import time
from typing import Any

import aiohttp
import ray
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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


def launch_router_process(
    worker_urls: list[str],
):
    router_ip = ray.util.get_node_ip_address().strip("[]")
    router_port, _ = get_free_port(router_ip)
    router_address = (
        f"[{router_ip}]:{router_port}" if is_valid_ipv6_address(router_ip) else f"{router_ip}:{router_port}"
    )

    router_process = multiprocessing.Process(
        target=run_router,
        args=(
            router_ip,
            router_port,
            worker_urls,
        ),
    )
    router_process.daemon = True
    router_process.start()
    time.sleep(3)
    assert router_process.is_alive()

    logger.info(f"Router is running on {router_address}")
    return router_address, router_process


def run_router(router_ip: str, router_port: int, worker_urls: list[str]):
    router = NaiveRouter(worker_urls=worker_urls, verbose=False)
    uvicorn.run(router.app, host=router_ip, port=router_port, log_level="warning")


class NaiveRouter:
    def __init__(
        self,
        worker_urls: list[str],
        max_connections: int = 1024,
        timeout: int = 60,
        max_attempts: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = False,
    ) -> None:
        """A minimal async load-balancing router."""
        self.verbose = verbose
        self.app = FastAPI()
        self.worker_urls = worker_urls
        self.request_counts = {url: 0 for url in worker_urls}

        self.max_connections = max_connections
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

        self.app = FastAPI()

        # Register startup / shutdown hooks
        self.app.on_event("startup")(self._on_startup)
        self.app.on_event("shutdown")(self._on_shutdown)

        # Catch-all proxy route
        self.app.api_route("/{endpoint:path}", methods=["GET", "POST"])(self._make_async_request)

        # Placeholder for aiohttp client
        self.client = None

    async def _on_startup(self):
        """Initialize aiohttp client safely inside the event loop"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections // 4,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=None)
        self.client = aiohttp.ClientSession(connector=connector, timeout=timeout)
        if self.verbose:
            logger.info(f"[router] aiohttp client initialized with max_connections={self.max_connections}")

    async def _on_shutdown(self):
        """Gracefully close aiohttp client"""
        if self.client and not self.client.closed:
            await self.client.close()
            if self.verbose:
                logger.info("[router] aiohttp client closed")

    async def _make_async_request(self, request: Request, endpoint: str):
        """Proxy single request to a worker URL."""
        if not self.worker_urls:
            return JSONResponse(status_code=503, content={"error": "No available workers"})

        worker_url = self._select_worker()
        target_url = f"{worker_url}/{endpoint}"

        if self.verbose:
            logger.debug(f"[router] Forwarding request → {target_url}")

        # Copy request data
        body = await request.body()
        headers = dict(request.headers)

        for attempt in range(self.max_attempts):
            # Send request to worker
            try:
                async with self.client.request(request.method, target_url, data=body, headers=headers) as response:
                    response.raise_for_status()
                    output = await _read_async_response(response)
                    self._release_worker(worker_url)
                    return output
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

    def _select_worker(self) -> str:
        """Select the least-loaded worker (simple round-robin by request count)."""
        url = min(self.request_counts, key=self.request_counts.get)
        self.request_counts[url] += 1
        return url

    def _release_worker(self, url: str) -> None:
        """Mark worker as free after request completes."""
        self.request_counts[url] = max(0, self.request_counts[url] - 1)
