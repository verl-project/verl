"""Collector — unified collector interface combining Transport + Decoder.

Composes a ``Transport`` (data acquisition) and a ``Decoder`` (data
interpretation and store writing).  Both ``start()`` and ``stop()``
are synchronous — async logic is encapsulated internally so the
upper layer (e.g. Ray actor) can call them without ``await``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future

from verl.workers.rollout.llm_router.collectors.decoder.base import Decoder
from verl.workers.rollout.llm_router.collectors.transport.base import Transport

logger = logging.getLogger(__name__)


class Collector:
    """Unified collector — composes Transport + Decoder.

    The Collector owns the lifecycle: it starts the Transport on a
    dedicated event-loop thread, passing the Decoder's ``decode``
    method as the handler.  ``store_cls`` is derived from the Decoder
    so the registry can look it up.

    Args:
        transport: Transport instance (ZMQ, HTTP, etc.)
        decoder: Decoder instance (vLLM KV, vLLM Metrics, etc.)
    """

    def __init__(self, transport: Transport, decoder: Decoder) -> None:
        self._transport = transport
        self._decoder = decoder
        self._future: Future | None = None
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None

    # ── Derived attributes ─────────────────────────────────────────────

    @property
    def store_cls(self) -> type:
        """Store class — derived from the Decoder."""
        return self._decoder.store_cls

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the collector — launch event-loop thread and subscribe.

        Spawns a dedicated event-loop thread, starts the Transport's
        subscribe loop on it, passing the Decoder's decode method
        as the handler.  Synchronous — no ``await`` needed.
        """
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
        )
        self._loop_thread.start()

        self._future = asyncio.run_coroutine_threadsafe(
            self._transport.subscribe(self._decoder.decode),
            self._loop,
        )

    def stop(self) -> None:
        """Stop the collector — stop transport, then stop event-loop thread.

        Synchronous — blocks until all cleanup is complete.
        """
        # First stop the transport (it cancels its own tasks and closes sockets)
        self._transport.stop()

        # Cancel and await the main subscribe future
        if self._future is not None:
            self._future.cancel()
            try:
                self._future.result()
            except (asyncio.CancelledError, Exception) as exc:
                logger.debug("Error waiting for collector task to finish: %s", exc)
            self._future = None

        # Stop the event loop and join the thread
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
            self._loop_thread = None
