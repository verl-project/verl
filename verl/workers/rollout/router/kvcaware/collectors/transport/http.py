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

"""HTTPTransport — Prometheus HTTP polling transport.

Polls ``http://{address}/metrics`` for each endpoint at a fixed interval
and delivers response text to the handler callback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import httpx

from ...collectors.transport.base import Transport

logger = logging.getLogger(__name__)


class HTTPTransport(Transport):
    """HTTP polling transport — fetches Prometheus metrics from endpoints.

    Each endpoint is polled at ``interval`` via ``httpx.AsyncClient``.
    Response text is delivered to the handler callback for decoding.

    Args:
        endpoints: ``{node_id: ip:port}`` — each address polls
            ``http://{address}/metrics``.
        interval: Polling interval in seconds.
        http_timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        endpoints: dict[str, str],
        interval: float = 5.0,
        http_timeout: float = 10.0,
    ) -> None:
        self._endpoints: dict[str, str] = {nid: f"http://{addr}/metrics" for nid, addr in endpoints.items()}
        self._interval = interval
        self._http_timeout = http_timeout
        self._client: httpx.AsyncClient | None = None

    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Start the HTTP polling loop — delivers response text to handler."""
        self._client = httpx.AsyncClient(timeout=self._http_timeout, trust_env=False)
        try:
            while True:
                coros = {nid: self._client.get(url) for nid, url in self._endpoints.items()}
                responses = await asyncio.gather(*coros.values(), return_exceptions=True)
                for nid, resp in zip(coros.keys(), responses, strict=False):
                    if isinstance(resp, Exception):
                        continue  # failed node — handler falls back to defaults
                    if resp.status_code != 200:
                        logger.warning(f"Failed to fetch metrics from {nid}: HTTP {resp.status_code}")
                        continue
                    try:
                        handler(resp.text, nid)
                    except Exception as exc:
                        logger.debug(f"Handler error for node {nid}: {exc}")
                await asyncio.sleep(self._interval)
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            client, self._client = self._client, None
            if client is not None:
                try:
                    await client.aclose()
                except Exception as exc:
                    # May fail if called outside an async context (e.g. GC finalizer)
                    logger.debug(f"HTTPTransport: aclose failed during cleanup: {exc}")

    def stop(self) -> None:
        """No protocol-level resources to close here.

        Per the Transport contract, task cancellation is owned by the
        ``Collector``. The httpx client's ``aclose()`` runs in
        ``subscribe``'s finally block, drained by ``Collector``'s
        ``_cancel_and_drain`` — nothing for this method to do.
        """
