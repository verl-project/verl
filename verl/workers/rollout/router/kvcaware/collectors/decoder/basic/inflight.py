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

"""InflightDecoder — Balancer callback → MetricsUpdate deltas (inflight ±1).

Also dispatch/complete counts, dispatch turn, and dispatched prompt length.

Mirrors verl ``GlobalRequestLoadBalancer._inflight_requests``: acquire bumps
the chosen replica's in-flight count by +1, release decrements it by -1. The
decoder is stateless — it emits only signed deltas; the store's ``incr`` owns
the running counters.

Beyond the running inflight gauge, acquire/release also bump the per-replica
cumulative DISPATCHED_COUNT / COMPLETED_COUNT counters — the monotonic
siblings of the gauge that carry dispatch / completion volume per replica.
Summed across replicas they give the global dispatched/completed totals; their
ratio is the realized-throughput share. On acquire the decoder also forwards
``request_id``: the collector uses it to bump the per-request turn counter
(``PerRequestStore``) and attribute that turn to the receiving replica's
``TURN_SUM`` — so the per-replica view and the turn-weighted view share the
inflight collector's home (one callback subscription, one decode pass). On
acquire the request's input prompt length (``event.prompt_len``) is also
forwarded as a ``PROMPT_LEN_SUM`` delta — the per-replica cumulative
request-size signal the plot derives an average dispatched prompt length from.

``INFLIGHT_TOKENS`` is the token-weighted sibling of the inflight gauge: acquire
adds the request's ``prompt_len`` and release subtracts it. acquire/release run
in the same ``generate()`` scope and see the same ``prompt_ids``, so the balancer
forwards the identical ``prompt_len`` on both — the gauge is symmetric and needs
no request→len bookkeeping in the store.

Event → delta mapping:
- ``on_acquire`` → ``INFLIGHT_COUNT``/``DISPATCHED_COUNT`` +1,
  ``INFLIGHT_TOKENS``/``PROMPT_LEN_SUM`` +``prompt_len``, carrying ``request_id``.
- ``on_release`` → ``INFLIGHT_COUNT`` -1, ``INFLIGHT_TOKENS`` -``prompt_len``,
  ``COMPLETED_COUNT`` +1.

``on_servers_removed`` is intentionally a no-op (returns ``None``): verl never
removes servers and maintains ``_inflight_requests`` purely via symmetric
acquire/release. Clearing on removal would be unsafe — in-flight requests for
a removed replica still complete and fire their ``on_release`` (-1), which
would drive the counter negative if we'd zeroed it on removal. The cumulative
counts are likewise left intact (they are lifetime totals). Faithful
simulation = don't touch the counters on removal.
"""

from __future__ import annotations

from typing import Any

from ....collectors.decoder import Decoder, MetricsUpdate
from ....collectors.transport.callback import StatisticEvent
from ....types import MetricKey


class InflightDecoder(Decoder):
    """Decode ``StatisticEvent`` → inflight + dispatch/complete ``MetricsUpdate`` deltas."""

    def decode(self, raw_data: bytes | str | Any, node_id: str) -> MetricsUpdate | None:
        """Dispatch on acquire/release; ignore other events / non-event payloads."""
        if not isinstance(raw_data, StatisticEvent):
            return None

        event = raw_data
        if event.replica_id is None:
            return None

        if event.event == "on_acquire":
            return MetricsUpdate(
                node_id=event.replica_id,
                metrics={
                    MetricKey.INFLIGHT_COUNT: 1,
                    MetricKey.INFLIGHT_TOKENS: event.prompt_len,
                    MetricKey.DISPATCHED_COUNT: 1,
                    MetricKey.PROMPT_LEN_SUM: event.prompt_len,
                },
                is_delta=True,
                request_id=event.request_id,
            )
        if event.event == "on_release":
            return MetricsUpdate(
                node_id=event.replica_id,
                metrics={
                    MetricKey.INFLIGHT_COUNT: -1,
                    MetricKey.INFLIGHT_TOKENS: -event.prompt_len,
                    MetricKey.COMPLETED_COUNT: 1,
                },
                is_delta=True,
            )
        return None
