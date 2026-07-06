"""ZMQTransport — ZMQ replay + sub dual socket transport.

Connects to per-replica ZMQ endpoints (sub + replay), subscribes to
live events, and delivers raw payloads to the handler callback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

import zmq
import zmq.asyncio

from verl.workers.rollout.llm_router.collectors.transport.base import Transport

logger = logging.getLogger(__name__)


@dataclass
class _ReplicaSocketSet:
    """Per-replica ZMQ socket bundle (internal — not exported)."""

    node_id: str
    context: zmq.asyncio.Context
    sub_socket: zmq.asyncio.Socket
    replay_socket: zmq.asyncio.Socket


class ZMQTransport(Transport):
    """ZMQ transport — replay + sub dual socket per replica.

    Each replica endpoint gets its own ZMQ context, socket pair, and
    background coroutine — all replicas subscribe concurrently.

    Args:
        endpoints: ``{replica_id: [sub_ip:port, replay_ip:port]}``
            — per-replica ZMQ endpoint addresses.
        topic: ZMQ subscription topic filter (default ``"kv-events"``).
        base_retry_delay: Initial retry delay in seconds.
        max_retry_delay: Maximum retry delay cap in seconds.
        max_retry_attempts: Maximum number of retries per replica.
        retry_backoff_factor: Exponential backoff multiplier.
    """

    def __init__(
        self,
        endpoints: dict[str, list[str]],
        topic: str = "kv-events",
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        max_retry_attempts: int = 5,
        retry_backoff_factor: float = 2.0,
    ) -> None:
        self._topic = topic
        self._base_retry_delay = base_retry_delay
        self._max_retry_delay = max_retry_delay
        self._max_retry_attempts = max_retry_attempts
        self._retry_backoff_factor = retry_backoff_factor

        self._sub_endpoints: dict[str, str] = {}
        self._replay_endpoints: dict[str, str] = {}
        for replica_id, addrs in endpoints.items():
            self._sub_endpoints[replica_id] = f"tcp://{addrs[0]}"
            self._replay_endpoints[replica_id] = f"tcp://{addrs[1]}"

        self._stopped = False
        self._replica_sockets: dict[str, _ReplicaSocketSet] = {}
        self._retry_counts: dict[str, int] = {}
        self._sub_tasks: dict[str, asyncio.Task] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Spawn per-replica subscription tasks, deliver payloads to handler."""
        self._loop = asyncio.get_running_loop()
        sub_tasks = []
        for node_id in self._sub_endpoints:
            sub_addr = self._sub_endpoints[node_id]
            replay_addr = self._replay_endpoints[node_id]
            t = asyncio.create_task(self._subscribe_for_replica(node_id, sub_addr, replay_addr, handler))
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
        """Stop ZMQ subscription synchronously — blocks until cleanup is done."""
        self._stopped = True
        for task in self._sub_tasks.values():
            task.cancel()
        for task in self._sub_tasks.values():
            try:
                if self._loop is not None:
                    done_future = asyncio.run_coroutine_threadsafe(
                        self._wait_task(task),
                        self._loop,
                    )
                    done_future.result(timeout=10)
            except (asyncio.CancelledError, Exception) as exc:
                logger.debug("Error waiting for ZMQ sub task to finish: %s", exc)
        self._sub_tasks.clear()
        self._close_all_zmq_sockets()

    async def _wait_task(self, task: asyncio.Task) -> None:
        """Await a task — used by stop() for blocking wait."""
        try:
            await task
        except asyncio.CancelledError:
            pass

    # ── ZMQ connection management ───────────────────────────────────────

    async def _connect_zmq_for(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
    ) -> bool:
        """Create ZMQ context and replay + sub dual socket for a single replica."""
        try:
            ctx = zmq.asyncio.Context()

            sub_socket = ctx.socket(zmq.SUB)
            sub_socket.connect(sub_addr)
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)

            replay_socket = ctx.socket(zmq.REQ)
            replay_socket.connect(replay_addr)

            self._replica_sockets[node_id] = _ReplicaSocketSet(
                node_id=node_id,
                context=ctx,
                sub_socket=sub_socket,
                replay_socket=replay_socket,
            )
            self._retry_counts[node_id] = 0
            return True

        except zmq.ZMQError as exc:
            logger.warning("ZMQ connection error for node %s: %s", node_id, exc)
            self._close_zmq_sockets_for(node_id)
            return False

    def _close_zmq_sockets_for(self, node_id: str) -> None:
        """Safely close ZMQ sockets and context for a single replica."""
        sockets = self._replica_sockets.pop(node_id, None)
        if sockets is None:
            return
        sockets.sub_socket.close()
        sockets.replay_socket.close()
        sockets.context.term()

    def _close_all_zmq_sockets(self) -> None:
        """Close all per-replica ZMQ sockets and contexts."""
        for node_id in list(self._replica_sockets.keys()):
            self._close_zmq_sockets_for(node_id)

    async def _reconnect_with_backoff_for(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
    ) -> bool:
        """Exponential backoff reconnect for a single replica."""
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

    # ── Per-replica subscription ────────────────────────────────────────

    async def _subscribe_for_replica(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
        handler: Callable[[bytes | str, str], None],
    ) -> None:
        """Per-replica subscription: connect → replay → subscribe loop."""
        try:
            if not await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                    return

            await self._replay_historical_data_for(node_id, handler)

            sockets = self._replica_sockets.get(node_id)
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
        """Request replay of historical data for a single replica.
        Degrade to subscription-only on failure."""
        sockets = self._replica_sockets.get(node_id)
        if sockets is None or sockets.replay_socket is None:
            return

        try:
            await sockets.replay_socket.send(b"replay")

            try:
                replay_data = await asyncio.wait_for(
                    sockets.replay_socket.recv(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                return  # timeout → degrade to subscription-only

            if replay_data:
                for line in replay_data.splitlines():
                    if line.strip():
                        handler(line, node_id)

        except zmq.ZMQError as exc:
            logger.warning("ZMQ replay error for node %s: %s", node_id, exc)
