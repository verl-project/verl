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

"""Collector — unified collector interface combining Transport + Decoder."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from concurrent.futures import Future

from ..config.collector import CollectorConfig
from ..logging import get_router_logger
from ..store.data_store import DataStore
from ..types import MetricKey
from .decoder import Decoder, KVCacheUpdate, MetricsUpdate, StickyUpdate
from .transport.base import Transport

logger = get_router_logger("collector")

# Log polled Prometheus metrics every N metrics-writes (≈every 10 s at the
# default 1 s polling interval × a few replicas). Lets us compare what the
# collector feeds the router against vllm's own engine-stats log.
_METRICS_LOG_EVERY_POLLS = 30

# Emit the per-replica dispatched/completed/turn_sum snapshot (the
# `router-dispatch` log line) at most this often (seconds). Time-throttled so
# the cadence is load-independent; checked on each acquire/release, so idle
# stretches emit nothing.
_DISPATCH_LOG_INTERVAL_S = 5.0

# Cumulative metrics tracked for windowed deltas in the evidence log. Single
# source of truth — ``_delta`` consumers below read these by key, and the
# per-replica prev-snapshot iterates the same tuple.
_CUMULATIVE_KEYS: tuple[str, ...] = (
    MetricKey.TTFT_SECONDS_SUM,
    MetricKey.TTFT_COUNT,
    MetricKey.QUEUE_TIME_SECONDS_SUM,
    MetricKey.QUEUE_TIME_COUNT,
    MetricKey.TPOT_SECONDS_SUM,
    MetricKey.TPOT_COUNT,
    MetricKey.PROMPT_TOKENS,
    MetricKey.PROMPT_TOKENS_CACHED,
    MetricKey.GENERATION_TOKENS,
    MetricKey.EXTERNAL_PREFIX_CACHE_HITS,
    MetricKey.ESTIMATED_FLOPS_PER_GPU,
)


def _avg(delta_sum: float, delta_cnt: float) -> float:
    """Windowed average = delta_sum / delta_cnt, or NaN if no samples."""
    return delta_sum / delta_cnt if delta_cnt > 0 else float("nan")


def _ms(value: float) -> str:
    """Format a seconds value as millis for the evidence log ('-' if NaN)."""
    return f"{value * 1000:.1f}" if value == value else "-"


# Log a kv-events tally (events + blocks by type) every N applied updates.
# Lets us see BlockStored/BlockRemoved flow — esp. whether mc-off groups (no
# mooncake → no kv-events emission) get any events at all.
_KV_EVENT_LOG_EVERY = 500


class Collector:
    """Unified collector — composes Transport + Decoder.

    Args:
        transport: Transport instance (ZMQ, HTTP, etc.)
        decoder: Decoder instance (vLLM KV, vLLM Metrics, etc.)
    """

    def __init__(self, transport: Transport, decoder: Decoder) -> None:
        self._transport = transport
        self._decoder = decoder
        self._data_store = DataStore()
        self._future: Future | None = None
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None
        # Periodic evidence-log bookkeeping (metrics decoder only). The
        # decoder itself is stateless — it returns MetricsUpdate and we merge
        # it here, so the merged-store snapshot the log reads is current.
        self._metrics_poll_count = 0
        # Previous cumulative snapshot per node — for windowed delta
        # computation. {node_id: {canonical_key: value}}
        self._metrics_prev: dict[str, dict[str, float]] = {}
        # kv-event tallies for periodic summary logging (kv decoder only).
        self._kv_event_counts: dict[str, int] = defaultdict(int)
        self._kv_block_counts: dict[str, int] = defaultdict(int)
        self._kv_last_logged_total = 0
        # Last-emit time for the dispatched/completed/turn_sum snapshot (throttled).
        self._dispatch_last_log: float = 0.0

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the collector — launch event-loop thread and subscribe."""

        def run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        def handler(raw_data: bytes | str, node_id: str) -> None:
            """Handler: decode and dispatch to the right store write path."""
            result = self._decoder.decode(raw_data, node_id)
            if isinstance(result, KVCacheUpdate):
                self._write_kv_update(result)
            elif isinstance(result, MetricsUpdate):
                self._write_metrics_update(result)
            elif isinstance(result, StickyUpdate):
                self._write_sticky_update(result)
            else:
                # None is normal for statistic decoders that skip an event
                # (e.g. StickyDecoder on_release); demote to debug to avoid per-turn noise.
                logger.debug(f"decoder.decode returned no update: {result}")

        if getattr(self._transport, "is_async", True):
            self._loop_thread = threading.Thread(
                target=run_loop,
                daemon=True,
            )
            self._loop_thread.start()

            self._future = asyncio.run_coroutine_threadsafe(
                self._transport.subscribe(handler),
                self._loop,
            )
        else:
            # CallbackTransport: subscribe only registers callbacks — run it on a
            # throwaway loop; stop() just unregisters.
            tmp_loop = asyncio.new_event_loop()
            try:
                tmp_loop.run_until_complete(self._transport.subscribe(handler))
            finally:
                tmp_loop.close()

    def _write_kv_update(self, update: KVCacheUpdate) -> None:
        """Write KVCacheUpdate via DataStore, then emit a periodic kv-events tally."""
        if update.block_size is not None:
            self._data_store.set_block_size(update.block_size)
        if update.clear_all:
            self._data_store.clear_kv_node(update.node_id)
        for layer, hashes in update.remove_blocks.items():
            if hashes:
                self._data_store.remove_kv_blocks(update.node_id, hashes, layer=layer)
        for layer, hashes in update.add_blocks.items():
            if hashes:
                self._data_store.add_kv_blocks(update.node_id, hashes, layer=layer)

        # Tally for periodic summary — observe BlockStored/BlockRemoved flow.
        n_added = sum(len(v) for v in update.add_blocks.values())
        n_removed = sum(len(v) for v in update.remove_blocks.values())
        if update.clear_all:
            self._kv_event_counts["clear"] += 1
        if n_added:
            self._kv_event_counts["stored"] += 1
            self._kv_block_counts["stored"] += n_added
        if n_removed:
            self._kv_event_counts["removed"] += 1
            self._kv_block_counts["removed"] += n_removed
        total = sum(self._kv_event_counts.values())
        if total - self._kv_last_logged_total >= _KV_EVENT_LOG_EVERY:
            self._kv_last_logged_total = total
            logger.info(
                f"kv-events tally: events={dict(self._kv_event_counts)} "
                f"blocks={dict(self._kv_block_counts)} (total_events={total}) | "
                f"retained_blocks/replica={self._data_store.per_replica_block_counts()}"
            )

    def _write_metrics_update(self, update: MetricsUpdate) -> None:
        """Write MetricsUpdate via DataStore, then emit a periodic evidence log.

        Delta updates route to ``incr_metric``; a dispatch (DISPATCHED_COUNT
        bumped) also records its turn to ``TURN_SUM``. Both acquire and release
        refresh the throttled ``router-dispatch`` snapshot. Absolute (non-delta)
        updates are polled gauges → evidence-log path below.
        """
        if update.is_delta:
            # Batch the decoder's signed deltas in ONE locked PerReplica write.
            # An on_acquire update carries INFLIGHT/DISPATCHED/PROMPT_LEN_SUM;
            # a dispatch (DISPATCHED_COUNT bumped) also folds in TURN_SUM. Turn
            # fires only on a dispatch, not on request_id presence — a
            # request_id-carrying non-dispatch update must not be mis-counted.
            # The turn lookup runs first (it lives in PerRequestStore, a
            # separate lock) so its result can join this batch instead of
            # needing a second PerReplica lock cycle.
            deltas = dict(update.metrics)
            if MetricKey.DISPATCHED_COUNT in deltas:
                if update.request_id is None:
                    logger.debug("dispatch (DISPATCHED_COUNT) update missing request_id — skipping turn")
                else:
                    deltas[MetricKey.TURN_SUM] = self._data_store.incr_per_request(update.request_id, "turn")
            self._data_store.incr_metrics(update.node_id, deltas)
            self._maybe_log_dispatch_stats()
            return
        self._data_store.refresh_metrics({update.node_id: update.metrics})

        # Periodic visibility into what the collector fed the router — compare
        # against vllm's own "GPU KV cache usage" engine-stats log line.
        self._metrics_poll_count += 1
        if self._metrics_poll_count % _METRICS_LOG_EVERY_POLLS == 0:
            # Emit evidence for ALL known replicas, not just the one that happened
            # to be polled at this poll-count tick. Metrics polling is serial
            # (one replica per poll), so emitting only ``update.node_id`` here
            # sampled ~1/N of replicas per window → some replicas never got an
            # evidence line (e.g. 4/8 seen). Each replica keeps its own
            # ``_metrics_prev`` baseline, so windowed deltas stay correct.
            for nid in self._data_store.get_metric_node_ids():
                self._log_evidence_window(nid)

    def _write_sticky_update(self, update: StickyUpdate) -> None:
        """Apply a StickyUpdate to the per-request store (sticky key) via DataStore."""
        if update.action == "put":
            self._data_store.put_sticky_binding(update.request_id, update.replica_id)
        elif update.action == "invalidate":
            self._data_store.invalidate_sticky_binding(update.request_id)
        elif update.action == "invalidate_replica":
            for rid in update.replica_ids:
                self._data_store.invalidate_sticky_replica(rid)
        else:
            logger.warning(f"unknown StickyUpdate action: {update.action}")

    def _maybe_log_dispatch_stats(self) -> None:
        """Emit per-replica dispatched/completed/turn_sum/prompt_len_sum counters at most every interval.

        Reads each dispatched replica's cumulative counters from PerReplicaStore and
        logs them (the ``router-dispatch`` line); the plot derives trailing-5-min
        dispatched / completed / avg-turn / RPM / avg-prompt-len from their per-replica
        deltas. Time-throttled so the cadence is load-independent (idle stretches emit nothing).
        """
        now = time.monotonic()
        if now - self._dispatch_last_log < _DISPATCH_LOG_INTERVAL_S:
            return
        self._dispatch_last_log = now
        for rep in self._data_store.get_metric_node_ids():
            snap = self._data_store.get_metrics(rep)
            dispatched = snap.get(MetricKey.DISPATCHED_COUNT, 0)
            if not dispatched:  # skip replicas that never received a dispatch
                continue
            completed = snap.get(MetricKey.COMPLETED_COUNT, 0)
            turn_sum = snap.get(MetricKey.TURN_SUM, 0)
            prompt_len_sum = snap.get(MetricKey.PROMPT_LEN_SUM, 0)
            logger.info(
                f"router-dispatch replica={rep} dispatched={dispatched} completed={completed} "
                f"turn_sum={turn_sum} prompt_len_sum={prompt_len_sum}"
            )

    def _log_evidence_window(self, node_id: str) -> None:
        """Emit a windowed evidence summary for one replica.

        Computes deltas vs the previous snapshot for cumulative counters/
        histograms so each line is a rate/average over ~``_METRICS_LOG_EVERY_POLLS``
        polls (≈30 s at the default 1 s interval). This is the raw feed for the
        B−A / D−C evidence chain (TTFT↓, prompt_tokens↓, cached↑ for kvcare).

        Read from the merged store snapshot (refresh already happened above)
        rather than the per-poll ``update.metrics`` — a transiently-missing
        scrape line would otherwise zero a cumulative counter and corrupt the
        window delta.
        """
        snap = self._data_store.get_metrics(node_id)
        prev = self._metrics_prev.get(node_id, {})

        def _delta(key: str) -> float:
            cur = float(snap.get(key, 0) or 0)
            return cur - float(prev.get(key, cur) or 0)

        # kv = retained occupancy (retained_blocks/num_gpu_blocks) — the signal
        # the strategy's load formula routes on. usage = vLLM's KV_CACHE_USAGE_PERC,
        # the running-only fraction (1 - num_free_blocks/num_gpu_blocks; the free
        # pool includes cached-but-freeable blocks, so usage EXCLUDES the prefix
        # cache — vLLM docs: "1 means 100 percent usage"). Complementary signals:
        # kv = hash-bearing blocks (free-cached + running-with-hash), usage =
        # running-only. Emit both so the pressure story shows cache-fill (kv) vs
        # running-pressure (usage) — eviction churns the cached-freeable sliver as
        # usage → 1, which is why evictions climb before kv reaches 1.0.
        kv = self._data_store.kv_cache_load(node_id)
        usage_raw = snap.get(MetricKey.KV_CACHE_USAGE_PERC)
        run = snap.get(MetricKey.NUM_REQUESTS_RUNNING)
        wait = snap.get(MetricKey.NUM_REQUESTS_WAITING)

        # Windowed TTFT/queue/TPOT averages (delta_sum / delta_count).
        ttft_avg = _avg(_delta(MetricKey.TTFT_SECONDS_SUM), _delta(MetricKey.TTFT_COUNT))
        queue_avg = _avg(_delta(MetricKey.QUEUE_TIME_SECONDS_SUM), _delta(MetricKey.QUEUE_TIME_COUNT))
        # prefill_time = TTFT - queue_wait. TTFT includes queue; subtracting it
        # isolates the real prefill compute cost that prefix-sharing reduces.
        prefill_t = (ttft_avg - queue_avg) if (ttft_avg == ttft_avg and queue_avg == queue_avg) else float("nan")
        tpot_avg = _avg(_delta(MetricKey.TPOT_SECONDS_SUM), _delta(MetricKey.TPOT_COUNT))

        # Token deltas over the window (prefill computed vs cached, decode, external).
        d_prefill = _delta(MetricKey.PROMPT_TOKENS)
        d_cached = _delta(MetricKey.PROMPT_TOKENS_CACHED)
        d_decode = _delta(MetricKey.GENERATION_TOKENS)
        d_external = _delta(MetricKey.EXTERNAL_PREFIX_CACHE_HITS)
        d_flops = _delta(MetricKey.ESTIMATED_FLOPS_PER_GPU)
        cache_hit_pct = 100.0 * d_cached / (d_cached + d_prefill) if (d_cached + d_prefill) > 0 else float("nan")

        kv_str = f"{kv:.3f}" if isinstance(kv, float) else kv
        usage_str = f"{float(usage_raw):.3f}" if usage_raw is not None else "-"
        hit_str = f"{cache_hit_pct:.1f}" if cache_hit_pct == cache_hit_pct else "-"
        logger.info(
            f"vllm-evidence replica={node_id} kv={kv_str} usage={usage_str} run={run} wait={wait} | "
            f"TTFT={_ms(ttft_avg)}ms queue={_ms(queue_avg)}ms prefillT={_ms(prefill_t)}ms TPOT={_ms(tpot_avg)}ms | "
            f"prefill={int(d_prefill)} cached={int(d_cached)} (hit={hit_str}%) "
            f"decode={int(d_decode)} external={int(d_external)} flops={int(d_flops)} [poll #{self._metrics_poll_count}]"
        )

        # Snapshot current cumulative values for next window's delta.
        self._metrics_prev[node_id] = {k: float(snap.get(k, 0) or 0) for k in _CUMULATIVE_KEYS}

    def stop(self) -> None:
        """
        Stop the collector — cancel tasks, drain cleanup, stop event-loop thread.
        """
        # Transport closes protocol-level resources (sockets/clients);
        # we own task cancellation and finally-block draining below.
        self._transport.stop()

        if self._loop.is_running():
            # Cancel all tasks and wait for their finally blocks inside the loop
            # so that aclose() runs while the loop is still alive.
            async def _cancel_and_drain() -> None:
                current = asyncio.current_task()
                tasks = [t for t in asyncio.all_tasks() if not t.done() and t is not current]
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            drain = asyncio.run_coroutine_threadsafe(_cancel_and_drain(), self._loop)
            try:
                drain.result(timeout=15)
            except Exception as exc:
                logger.debug(f"Error draining tasks on stop: {exc}")

            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
            self._loop_thread = None

        self._future = None


# ── Factory function ───────────────────────────────────────────────────


def get_collector(
    name: str,
    collectors_config: CollectorConfig,
    server_addresses: dict[str, str] | None = None,
    kv_event_endpoints: dict[str, list[str]] | None = None,
    balancer_handler=None,
) -> Collector:
    """Create a Collector by name — one place does both composition and config binding.

    Args:
        name: Collector type — ``"vllm_metrics"`` or ``"vllm_zmq"``.
        collectors_config: ``CollectorConfig`` carrying connection-type knobs.
        server_addresses: ``{node_id: ip:port}`` for HTTP transport
            (used by ``"vllm_metrics"``).
        kv_event_endpoints: ``{node_id: [sub_addr, replay_addr]}`` for ZMQ
            transport (used by ``"vllm_zmq"``).

    Returns:
        Configured ``Collector`` instance.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "vllm_metrics":
        from .decoder.vllm.metrics import VLLMMetricsDecoder
        from .transport.http import HTTPTransport

        hp = collectors_config.http_polling
        transport = HTTPTransport(
            endpoints=server_addresses or {},
            interval=hp["polling_interval"],
            http_timeout=hp["http_timeout"],
        )
        return Collector(transport, VLLMMetricsDecoder())

    if name == "vllm_zmq":
        from .decoder.vllm.kv import VLLMKVDecoder
        from .transport.zmq import ZMQTransport

        lc = collectors_config.long_connection
        transport = ZMQTransport(
            endpoints=kv_event_endpoints or {},
            base_retry_delay=lc["base_retry_delay"],
            max_retry_delay=lc["max_retry_delay"],
            max_retry_attempts=lc["max_retry_attempts"],
            retry_backoff_factor=lc["retry_backoff_factor"],
        )
        return Collector(transport, VLLMKVDecoder())

    if name == "sticky_stat":
        from .decoder.basic.sticky import StickyDecoder
        from .transport.callback import CallbackTransport

        return Collector(CallbackTransport(balancer_handler), StickyDecoder())

    if name == "inflight_stat":
        from .decoder.basic.inflight import InflightDecoder
        from .transport.callback import CallbackTransport

        return Collector(CallbackTransport(balancer_handler), InflightDecoder())

    raise ValueError(
        f"Unknown collector: '{name}'. Available: ['vllm_metrics', 'vllm_zmq', 'sticky_stat', 'inflight_stat']"
    )
