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

"""CallbackTransport — Balancer callback transport (pure forwarder).

The network transports (HTTP/ZMQ) fetch data off the wire on a background
event loop. The Balancer's sticky/inflight state, by contrast, is produced by
its OWN request-path callbacks (``on_acquire`` / ``on_release`` /
``on_servers_removed``) — there is nothing to poll. ``CallbackTransport`` is
the isomorphic Transport for that source: ``is_async = False``, ``subscribe``
merely registers synchronous callbacks on the Balancer and returns, and
``stop`` unregisters them. It carries no domain logic — it packs each Balancer
callback into a ``StatisticEvent`` (see below) and forwards it to the handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ...collectors.transport.base import Transport


@dataclass(frozen=True)
class StatisticEvent:
    """Packed Balancer callback — the payload the callback transport delivers.

    The three Balancer hook points do not share a stable ``node_id`` and differ
    in arity, so they cannot be forced into ``(raw, node_id)``:

    - ``on_acquire(request_id, chosen, prompt_ids)``  → two strings + a token list
    - ``on_release(server_id, prompt_len)``           → string + int, missing node_id
    - ``on_servers_removed(ids)``                     → list[str], not bytes/str

    Attributes:
        event: Hook name — ``"on_acquire"`` / ``"on_release"`` /
            ``"on_servers_removed"``.
        request_id: The routing request id (set on ``on_acquire``).
        replica_id: The chosen replica (``on_acquire``) or the released server
            (``on_release``) — unified under one field so one decoder can treat
            both as "the replica this event is about".
        server_ids: Removed server ids (``on_servers_removed``).
        prompt_len: Input prompt length (``len(prompt_ids)``; 0 when no prompt
            was forwarded). Set on ``on_acquire`` and, symmetrically, on
            ``on_release`` — acquire/release share the request's ``prompt_ids``
            in one ``generate()`` scope, so the same value flows to both and the
            inflight-token gauge (+prompt_len on acquire, -prompt_len on release)
            stays balanced without any request→len bookkeeping.
    """

    event: str
    request_id: str | None = None
    replica_id: str | None = None
    server_ids: tuple[str, ...] = ()
    prompt_len: int = 0


class CallbackTransport(Transport):
    """Pure-forwarder Transport backed by Balancer callbacks.

    Args:
        balancer_handler: The Balancer (or any object exposing
            ``register_call_back(event, fn)`` / ``un_register_call_back``).
            Passed through from ``CollectorManager`` (which receives it from
            the Balancer's ``_init_manager``).
    """

    is_async = False

    def __init__(self, balancer_handler: Any) -> None:
        self._balancer = balancer_handler
        self._registered: list[tuple[str, Callable]] = []

    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Register three callbacks on the Balancer; no loop, returns at once.

        Each callback packs its arguments into a ``StatisticEvent`` and hands
        it to ``handler`` with an empty ``node_id`` (the event carries its own
        ids; the sticky/inflight decoders ignore ``node_id``).
        """
        balancer = self._balancer

        def _on_acquire(request_id: str, chosen: str, prompt_ids: list[int] | None = None) -> None:
            prompt_len = len(prompt_ids) if prompt_ids else 0
            handler(
                StatisticEvent(
                    "on_acquire",
                    request_id=request_id,
                    replica_id=chosen,
                    prompt_len=prompt_len,
                ),
                "",
            )

        def _on_release(server_id: str, prompt_len: int = 0) -> None:
            handler(StatisticEvent("on_release", replica_id=server_id, prompt_len=prompt_len), "")

        def _on_servers_removed(server_ids: list[str]) -> None:
            handler(StatisticEvent("on_servers_removed", server_ids=tuple(server_ids)), "")

        self._registered = [
            ("on_acquire", _on_acquire),
            ("on_release", _on_release),
            ("on_servers_removed", _on_servers_removed),
        ]
        for event, fn in self._registered:
            balancer.register_call_back(event, fn)

    def stop(self) -> None:
        """Unregister every callback registered by ``subscribe`` (idempotent)."""
        for event, fn in self._registered:
            self._balancer.un_register_call_back(event, fn)
        self._registered = []
