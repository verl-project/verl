"""Transport — abstract base for data transport layers.

A Transport fetches raw data from a network source (ZMQ, HTTP, etc.)
and delivers it to a handler callback.  It does NOT decode or interpret
the data — that is the Decoder's job.

Lifecycle:
    ``subscribe(handler)`` — async; starts the data-fetch loop,
        calls ``handler(raw_data, node_id)`` for each received item.
    ``stop()`` — sync; blocks until the transport has fully shut down.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class Transport(ABC):
    """Abstract base for data transport layers.

    Subclasses implement ``subscribe()`` with their protocol-specific
    connection and data-fetch logic.  ``stop()`` cancels connections
    and blocks until cleanup is complete.
    """

    @abstractmethod
    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Start data acquisition and deliver each item to handler.

        Args:
            handler: Callback that receives (raw_data, node_id).
                raw_data is ``bytes`` (ZMQ) or ``str`` (HTTP response text).
                node_id identifies the source replica/node.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the transport synchronously — blocks until cleanup is done."""
