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

"""Transport — abstract base for data transport layers.

A Transport fetches raw data from a network source (ZMQ, HTTP, etc.)
and delivers it to a handler callback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class Transport(ABC):
    """Abstract base for data transport layers.

    Subclasses implement ``subscribe()`` with their protocol-specific
    connection and data-fetch logic.  ``stop()`` cancels connections
    and blocks until cleanup is complete.

    Attributes:
        is_async: ``True`` (default) when ``subscribe`` runs a long-lived
            async loop the ``Collector`` must schedule on a background event
            loop. ``CallbackTransport`` sets it ``False`` — its ``subscribe``
            merely registers synchronous callbacks and returns immediately, so
            the ``Collector`` skips the loop/thread setup.
    """

    is_async: bool = True

    @abstractmethod
    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Start data acquisition and deliver each item to handler.

        Args:
            handler: Callback that receives (raw_data, node_id).
                raw_data is ``bytes`` (ZMQ) / ``str`` (HTTP response text),
                or a ``StatisticEvent`` (callback transport).
                node_id identifies the source endpoint/node (empty for the
                callback transport — the event carries its own ids).
        """

    @abstractmethod
    def stop(self) -> None:
        """Signal stop and close protocol-level resources (sockets/clients).

        Implementations should only:
          1. set a stop flag so subscribe loops exit,
          2. close sockets / contexts / http clients (idempotently).
        """
