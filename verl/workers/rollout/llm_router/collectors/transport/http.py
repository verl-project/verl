"""HTTPTransport — Prometheus HTTP polling transport.

Polls ``http://{address}/metrics`` for each replica at a fixed interval
and delivers response text to the handler callback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import httpx

from verl.workers.rollout.llm_router.collectors.transport.base import Transport

logger = logging.getLogger(__name__)


class HTTPTransport(Transport):
    """HTTP polling transport — fetches Prometheus metrics from replicas.

    Each replica endpoint is polled at ``interval`` via ``httpx.AsyncClient``.
    Response text is delivered to the handler callback for decoding.

    Args:
        endpoints: ``{replica_id: ip:port}`` — each address polls
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
        self._loop: asyncio.AbstractEventLoop | None = None

    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Start the HTTP polling loop — delivers response text to handler."""
        self._loop = asyncio.get_running_loop()
        # trust_env=False: bypass HTTP(S)_PROXY env vars so polling requests to
        # replica /metrics endpoints (e.g. 172.17.x or 127.0.0.1) are not hijacked
        # by a container's outbound proxy — that would blind the router.
        self._client = httpx.AsyncClient(timeout=self._http_timeout, trust_env=False)
        try:
            while True:
                coros = {nid: self._client.get(url) for nid, url in self._endpoints.items()}
                responses = await asyncio.gather(*coros.values(), return_exceptions=True)
                for nid, resp in zip(coros.keys(), responses, strict=False):
                    if isinstance(resp, Exception):
                        continue  # failed node — handler falls back to defaults
                    try:
                        handler(resp.text, nid)
                    except Exception as exc:
                        logger.debug("Handler error for node %s: %s", nid, exc)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Stop HTTP polling — close the client on the event-loop thread.

        The Collector cancels the subscribe task, which exits the loop.  The
        AsyncClient must be closed *on the running loop* (httpx/anyio raise
        ``NoEventLoopError`` if aclose runs after the loop stops), so we
        schedule the close via ``run_coroutine_threadsafe`` while the loop is
        still alive.
        """
        if self._client is not None and self._loop is not None and self._loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(self._client.aclose(), self._loop)
                fut.result(timeout=5)
            except Exception as exc:
                logger.debug("Error closing HTTP client: %s", exc)
            self._client = None
