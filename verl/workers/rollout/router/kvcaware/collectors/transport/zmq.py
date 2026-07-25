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

"""ZMQTransport — ZMQ replay + sub dual socket transport.

Connects to per-endpoint ZMQ addresses (sub + replay), subscribes to
live events, and delivers raw payloads to the handler callback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

import zmq
import zmq.asyncio

from ...collectors.transport.base import Transport
from ...logging import get_router_logger

logger = get_router_logger("zmq-transport")


@dataclass
class _EndpointSocketSet:
    """Per-endpoint ZMQ socket bundle (internal — not exported)."""

    node_id: str
    context: zmq.asyncio.Context
    sub_socket: zmq.asyncio.Socket
    replay_socket: zmq.asyncio.Socket | None
    closed: bool = False


class ZMQTransport(Transport):
    """ZMQ transport — replay + sub dual socket per endpoint.

    Each endpoint gets its own ZMQ context, socket pair, and
    background coroutine — all endpoints subscribe concurrently.

    Args:
        endpoints: ``{node_id: [sub_ip:port, replay_ip:port, publisher, topic]}``
        base_retry_delay: Initial retry delay in seconds.
        max_retry_delay: Maximum retry delay cap in seconds.
        max_retry_attempts: Maximum number of retries per endpoint.
        retry_backoff_factor: Exponential backoff multiplier.
    """

    def __init__(
        self,
        endpoints: dict[str, list[str]],
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        max_retry_attempts: int = 5,
        retry_backoff_factor: float = 2.0,
    ) -> None:
        self._base_retry_delay = base_retry_delay
        self._max_retry_delay = max_retry_delay
        self._max_retry_attempts = max_retry_attempts
        self._retry_backoff_factor = retry_backoff_factor

        # endpoints: {node_id: [sub_endpoint, replay_endpoint, publisher, topic]}
        # replay_endpoint may be empty ("") for sub-only mode (e.g. standalone
        # collector that has no Ray RPC to discover the replay socket).
        self._sub_endpoints: dict[str, str] = {}
        self._replay_endpoints: dict[str, str] = {}
        self._topics: dict[str, str] = {}
        for node_id, addrs in endpoints.items():
            if len(addrs) < 4:
                raise ValueError(
                    f"endpoint '{node_id}' needs 4 elements [sub, replay, publisher, topic], got {len(addrs)}"
                )
            if addrs[2] != "zmq":
                raise ValueError(f"endpoint '{node_id}' publisher must be 'zmq', got '{addrs[2]}'")
            self._sub_endpoints[node_id] = f"tcp://{addrs[0]}"
            self._replay_endpoints[node_id] = f"tcp://{addrs[1]}" if addrs[1] else ""
            self._topics[node_id] = addrs[3]

        self._stopped = False
        self._endpoint_sockets: dict[str, _EndpointSocketSet] = {}
        self._retry_counts: dict[str, int] = {}
        self._sub_tasks: dict[str, asyncio.Task] = {}

    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Spawn per-endpoint subscription tasks, deliver payloads to handler."""
        sub_tasks = []
        for node_id in self._sub_endpoints:
            sub_addr = self._sub_endpoints[node_id]
            replay_addr = self._replay_endpoints[node_id]
            t = asyncio.create_task(self._subscribe_for_endpoint(node_id, sub_addr, replay_addr, handler))
            self._sub_tasks[node_id] = t
            sub_tasks.append(t)
        try:
            await asyncio.gather(*sub_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in self._sub_tasks.values():
                t.cancel()
        finally:
            self._close_all_zmq_sockets()

    def stop(self) -> None:
        """Signal stop, cancel tasks, close ZMQ sockets. No loop dependency.

        This method only include:
          1. sets the stop flag so subscribe loops exit,
          2. cancels tasks (synchronous, no loop ref needed),
          3. closes sockets/contexts (idempotent via ``closed`` guard).
        """
        self._stopped = True
        for task in self._sub_tasks.values():
            task.cancel()
        self._sub_tasks.clear()
        self._close_all_zmq_sockets()

    # ── ZMQ connection management ───────────────────────────────────────

    async def _connect_zmq_for(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
    ) -> bool:
        """Create ZMQ context and replay + sub dual socket for a single endpoint.

        If ``replay_addr`` is empty (sub-only mode), the replay REQ socket is
        skipped — used by the standalone collector which has no replay endpoint.
        """
        try:
            ctx = zmq.asyncio.Context()

            sub_socket = ctx.socket(zmq.SUB)
            sub_socket.connect(sub_addr)
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, self._topics[node_id])

            replay_socket = None
            if replay_addr:
                replay_socket = ctx.socket(zmq.REQ)
                replay_socket.connect(replay_addr)

            self._endpoint_sockets[node_id] = _EndpointSocketSet(
                node_id=node_id,
                context=ctx,
                sub_socket=sub_socket,
                replay_socket=replay_socket,
            )
            self._retry_counts[node_id] = 0
            return True

        except zmq.ZMQError as exc:
            logger.warning(f"ZMQ connection error for node {node_id}: {exc}")
            self._close_zmq_sockets_for(node_id)
            return False

    def _close_zmq_sockets_for(self, node_id: str) -> None:
        """Safely close ZMQ sockets and context for a single endpoint."""
        sockets = self._endpoint_sockets.pop(node_id, None)
        if sockets is None or sockets.closed:
            return
        sockets.closed = True
        sockets.sub_socket.close(linger=0)
        if sockets.replay_socket is not None:
            sockets.replay_socket.close(linger=0)
        sockets.context.term()

    def _close_all_zmq_sockets(self) -> None:
        """Close all per-endpoint ZMQ sockets and contexts."""
        for node_id in list(self._endpoint_sockets.keys()):
            self._close_zmq_sockets_for(node_id)

    async def _reconnect_with_backoff_for(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
    ) -> bool:
        """Exponential backoff reconnect for a single endpoint."""
        retry_count = self._retry_counts.get(node_id, 0)
        while retry_count < self._max_retry_attempts:
            delay = min(
                self._base_retry_delay * (self._retry_backoff_factor**retry_count),
                self._max_retry_delay,
            )
            await asyncio.sleep(delay)
            retry_count += 1
            self._retry_counts[node_id] = retry_count

            if await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                return True

        return False

    # ── Per-endpoint subscription ────────────────────────────────────────

    async def _subscribe_for_endpoint(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
        handler: Callable[[bytes | str, str], None],
    ) -> None:
        """Per-endpoint subscription: connect → replay (if available) → subscribe loop."""
        try:
            if not await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                    return

            # Replay historical events only when a replay socket is connected
            # (sub-only mode skips this — standalone collector has no replay endpoint).
            if replay_addr:
                await self._replay_historical_data_for(node_id, handler)

            sockets = self._endpoint_sockets.get(node_id)
            if sockets is None:
                return

            while not self._stopped:
                try:
                    parts = await sockets.sub_socket.recv_multipart()
                    payload = parts[-1]
                    handler(payload, node_id)
                except zmq.ZMQError:
                    self._close_zmq_sockets_for(node_id)
                    if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                        break
                    if replay_addr:
                        await self._replay_historical_data_for(node_id, handler)

        except asyncio.CancelledError:
            pass
        finally:
            self._close_zmq_sockets_for(node_id)

    # ── Replay ──────────────────────────────────────────────────────────

    async def _replay_historical_data_for(
        self,
        node_id: str,
        handler: Callable[[bytes | str, str], None],
    ) -> None:
        """Request replay of historical data for a single endpoint.

        vLLM's ROUTER replay socket expects the starting sequence number as an
        8-byte big-endian integer — the REQ socket adds the empty delimiter frame
        itself, so the publisher receives ``[client_id, "", start_seq]``. It then
        streams each retained batch back as ``[seq_bytes, payload]`` frames,
        terminated by an empty-payload ``END_SEQ`` marker. We start from seq 0 to
        pull the whole retained buffer. Degrades to subscription-only on failure.
        """
        sockets = self._endpoint_sockets.get(node_id)
        if sockets is None or sockets.replay_socket is None:
            return

        try:
            await sockets.replay_socket.send((0).to_bytes(8, "big"))

            # Drain streamed replay frames until the empty-payload end marker
            # (or a timeout / socket error) — then degrade to subscription-only.
            while True:
                try:
                    frames = await asyncio.wait_for(
                        sockets.replay_socket.recv_multipart(),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    return  # no more frames → degrade to subscription-only

                # REQ delivers each publisher send as [seq_bytes, payload];
                # an empty payload marks end-of-replay (vLLM END_SEQ marker).
                if len(frames) < 2 or not frames[-1]:
                    return
                handler(frames[-1], node_id)

        except zmq.ZMQError as exc:
            logger.warning(f"ZMQ replay error for node {node_id}: {exc}")
