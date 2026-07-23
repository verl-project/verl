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

"""KVCache-aware runtime strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.strategy import KVCAwareStrategyConfig
from ..logging import get_router_logger
from ..types import Layer, MetricKey, OverloadMode, SlowCut
from .registry import StrategyRegistry

if TYPE_CHECKING:
    from ..store import DataStore
    from .base import ReplicaInfo

logger = get_router_logger("kvc-aware-strategy")

STICKY_TOP_SCORE = 1e9

DEFAULT_LOAD_WEIGHTS: tuple[float, float, float, float] = (0.5, 0.0, 0.0, 0.5)


class StrategyError(Exception):
    """Strategy construction or scoring error."""


class KVCacheAwareStrategy:
    """Runtime strategy constructed from a ``KVCAwareStrategyConfig``."""

    def __init__(
        self,
        *,
        alpha: float,
        load_threshold: float,
        layer_weights: dict[Layer, float],
        collector_names: list[str],
        weight: float,
        memory_overload_filter: bool = True,
        slow_cut: SlowCut | str = SlowCut.PREFIX_LOAD_AWARE,
        load_weights: tuple[float, float, float, float] = DEFAULT_LOAD_WEIGHTS,
        overload_mode: OverloadMode | str = OverloadMode.KV_LOAD,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise StrategyError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 < load_threshold < 1:
            raise StrategyError(f"load_threshold must be in (0, 1), got {load_threshold}")
        _valid_layers = {Layer.GPU, Layer.CPU, Layer.SSD}
        if set(layer_weights.keys()) != _valid_layers:
            raise StrategyError(f"layer_weights keys must be {_valid_layers}, got {set(layer_weights.keys())}")
        for layer_key, layer_weight in layer_weights.items():
            if layer_weight < 0:
                raise StrategyError(f"layer_weights[{layer_key}] must be >= 0, got {layer_weight}")
        weights_sum = sum(layer_weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise StrategyError(f"layer_weights values must sum to 1.0, got {weights_sum}")
        if not isinstance(memory_overload_filter, bool):
            raise StrategyError(f"memory_overload_filter must be a bool, got {memory_overload_filter!r}")
        try:
            slow_cut = SlowCut(slow_cut)
        except ValueError as exc:
            raise StrategyError(f"slow_cut must be one of {[m.value for m in SlowCut]}, got {slow_cut!r}") from exc
        try:
            overload_mode = OverloadMode(overload_mode)
        except ValueError as exc:
            raise StrategyError(
                f"overload_mode must be one of {[m.value for m in OverloadMode]}, got {overload_mode!r}"
            ) from exc
        if len(load_weights) != 4 or any(w < 0 for w in load_weights):
            raise StrategyError(f"load_weights must be 4 non-negative values, got {load_weights}")
        if abs(sum(load_weights) - 1.0) > 1e-6:
            raise StrategyError(f"load_weights must sum to 1.0, got {sum(load_weights)}")

        self.alpha = float(alpha)
        self.load_threshold = float(load_threshold)
        self.layer_weights = dict(layer_weights)
        self.collector_names = collector_names
        self.weight = weight
        self.memory_overload_filter = memory_overload_filter
        self.slow_cut = slow_cut
        self.load_weights = tuple(load_weights)
        self.overload_mode = overload_mode
        self._max_num_seqs: int | None = None
        self._max_num_batched_tokens: int | None = None
        logger.info(
            f"KVCacheAwareStrategy created: alpha={self.alpha:.2f}, "
            f"load_threshold={self.load_threshold:.2f}, load_weights={self.load_weights}, "
            f"memory_overload_filter={self.memory_overload_filter}, slow_cut={self.slow_cut.value}, "
            f"overload_mode={self.overload_mode.value}"
        )

    def set_capacity(self, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        """Inject ``--max-num-seqs`` from the server handle's rollout config."""
        if not isinstance(max_num_seqs, int) or max_num_seqs <= 0:
            raise StrategyError(f"max_num_seqs must be a positive int, got {max_num_seqs}")
        if not isinstance(max_num_batched_tokens, int) or max_num_batched_tokens <= 0:
            raise StrategyError(f"max_num_batched_tokens must be a positive int, got {max_num_batched_tokens}")
        self._max_num_seqs = max_num_seqs
        self._max_num_batched_tokens = max_num_batched_tokens
        logger.info(
            f"KVCacheAwareStrategy capacity set: max_num_seqs={max_num_seqs}"
            f"max_num_batched_tokens={max_num_batched_tokens}"
        )

    @classmethod
    def from_config(cls, cfg: KVCAwareStrategyConfig) -> KVCacheAwareStrategy:
        """Construct from config. ``max_num_seqs`` is injected by the Balancer
        via ``set_capacity`` after fetching from the server handle."""
        return cls(
            alpha=cfg.alpha,
            load_threshold=cfg.load_threshold,
            layer_weights=cfg.layer_weights,
            collector_names=cfg.collector_names,
            weight=cfg.weight,
            memory_overload_filter=cfg.memory_overload_filter,
            slow_cut=cfg.slow_cut,
            overload_mode=cfg.overload_mode,
        )

    def _compute_load(
        self,
        kv_usage: float,
        running: int | float,
        waiting: int | float,
        inflight: int | float = 0,
    ) -> float:
        """load = a·kv_usage + b·running/max + c·waiting/max + d·inflight/max (∈ [0,1], bigger = more loaded).

        Weights ``(a, b, c, d) = self.load_weights``; ``max = self._max_num_seqs``.
        ``inflight`` (the Balancer's own acquire/release counter, maintained
        synchronously) is the only term non-zero at cold start — the other three
        come from async-polled vLLM metrics, still 0 before the first poll lands.
        Its weight ``d`` keeps the first wave of requests from collapsing onto
        ``pool[0]`` when the polled terms are tied at 0.
        """
        if self._max_num_seqs is None:
            raise StrategyError("set_capacity() must be called before routing")
        a, b, c, d = self.load_weights
        max_num_seqs = self._max_num_seqs
        return (
            a * float(kv_usage)
            + b * min(1.0, float(running) / max_num_seqs)
            + c * min(1.0, float(waiting) / max_num_seqs)
            + d * min(1.0, float(inflight) / max_num_seqs)
        )

    def is_overloaded(
        self,
        store: DataStore,
        replica: ReplicaInfo,
    ) -> bool:
        """Return True if ``replica`` is overloaded (``load > load_threshold``).

        Used only by the sticky short-circuit to decide whether to send a
        returning session back to its bound replica. Combined scoring never
        consults overload.
        """
        if self.overload_mode == OverloadMode.NONE:
            return False
        if self.overload_mode == OverloadMode.KV_CACHE_USAGE_PERC:
            kv_perc = store.get_metric(replica.replica_id, MetricKey.KV_CACHE_USAGE_PERC) or 0.0
            logger.info(f"is-overload replica={replica.replica_id} kv_perc={kv_perc:.4f}")
            return kv_perc > self.load_threshold
        if self.overload_mode == OverloadMode.KV_LOAD:
            m = store.get_metrics(replica.replica_id)
            kv_usage = store.kv_cache_load(replica.replica_id)
            running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
            waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
            inflight = m.get(MetricKey.INFLIGHT_COUNT, 0)
            load = self._compute_load(kv_usage, running, waiting, inflight)
            # Emit the load the sticky check used (one replica — the bound one) so the
            # plot can show the overload-check load alongside the combined-score load.
            logger.info(f"is-overload replica={replica.replica_id} kv_load={load:.4f}")
            return load > self.load_threshold
        raise ValueError(
            f"There is no {self.overload_mode}, please set overload_mode in ['None', 'kv_cache_usage_perc', 'kv_load']"
        )

    def _sticky_shortcut(
        self,
        store: DataStore,
        replicas: list[ReplicaInfo],
        request_id: str | None,
    ) -> list[float] | None:
        """Return a pre-built score list if a sticky replica should win, else None.

        Sticky wins when ``request_id`` is provided and the bound replica (from
        ``store.get_sticky_binding``) is present in ``replicas``. When
        ``memory_overload_filter`` is set the bound replica must also NOT be
        overloaded; otherwise the overload check is skipped. On win, returns a
        list with ``STICKY_TOP_SCORE`` at the bound index and ``0.0`` elsewhere;
        else ``None`` (fall through).
        """
        if not request_id:
            return None
        sticky_id = store.get_sticky_binding(request_id)
        if sticky_id is None:
            return None
        for idx, replica in enumerate(replicas):
            if replica.replica_id == sticky_id:
                if self.is_overloaded(store, replica):
                    logger.info(f"score(): STICKY replica={sticky_id} OVERLOADED → fallback")
                    return None
                logger.info(f"score(): STICKY replica={sticky_id} HIT → short-circuit (top score)")
                scores = [0.0] * len(replicas)
                scores[idx] = STICKY_TOP_SCORE
                return scores
        logger.info(f"score(): sticky replica={sticky_id} not in pool → fallback")
        return None

    def score(
        self,
        prompt_ids: list[int] | None,
        store: DataStore,
        replicas: list[ReplicaInfo],
        request_id: str | None = None,
    ) -> list[float]:
        """Score each replica. Larger is better.

        After the sticky short-circuit misses, the ``slow_cut`` mode selects the
        fallback scoring: ``prefix-load-aware`` → ``S = α·S_cache + (1-α)·S_load``;
        ``least-inflight`` → ``-INFLIGHT_COUNT`` (verl GlobalRequestLoadBalancer-style).
        """
        if not isinstance(replicas, list):
            raise StrategyError(f"replicas must be a list, got {type(replicas).__name__}")
        if not replicas:
            return []
        # Sticky short-circuit.
        shortcut = self._sticky_shortcut(store, replicas, request_id)
        if shortcut is not None:
            return shortcut
        if self.slow_cut == SlowCut.LEAST_INFLIGHT:
            return [-store.get_metric(r.replica_id, MetricKey.INFLIGHT_COUNT) for r in replicas]
        if self.slow_cut == SlowCut.CAPACITY_TOKEN_AWARE:
            return self._capacity_token_scores(store, replicas, prompt_ids or [])
        effective_prompt_ids = prompt_ids or []

        result = []
        loads: dict[str, float] = {}
        for replica in replicas:
            m = store.get_metrics(replica.replica_id)
            kv_usage = store.kv_cache_load(replica.replica_id)
            running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
            waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
            inflight = m.get(MetricKey.INFLIGHT_COUNT, 0)
            load = self._compute_load(kv_usage, running, waiting, inflight)
            loads[replica.replica_id] = load
            s_load = 1.0 - load
            s_cache, gpu_hit = self._cache_score(store, replica, effective_prompt_ids)
            score = self.alpha * s_cache + (1 - self.alpha) * s_load
            result.append(score)
            logger.info(
                f"score(): replica={replica.replica_id} kv={kv_usage:.3f} running={running} waiting={waiting} "
                f"inflight={inflight} → load={load:.4f} s_load={s_load:.4f} | gpu_hit={gpu_hit:.2f} "
                f"s_cache={s_cache:.4f} ({self.alpha:.2f}·cache + {1 - self.alpha:.2f}·load) → score={score:.4f}"
            )
        scores_str = ", ".join(f"{r.replica_id}={result[i]:.4f}" for i, r in enumerate(replicas))
        logger.info(f"score(): COMBINED scores: {scores_str}")
        # Per-replica load that drove this combined-score dispatch (all replicas,
        # reused from the loop above — no extra computation). The plot parses this
        # into a per-replica load panel; sticky-win / least-inflight dispatches do
        # not reach here, so they emit no line (panel omits those dispatches).
        logger.info(f"route-load loads={loads}")
        return result

    def _cache_score(
        self,
        store: DataStore,
        replica: ReplicaInfo,
        prompt_ids: list[int],
    ) -> tuple[float, float]:
        """Three-layer weighted prefix-cache hit score ∈ [0, 1].

            S_cache = w_gpu·gpu_hit + w_cpu·cpu_hit + w_ssd·ssd_hit

        Each ``*_hit`` comes from ``get_layer_prefix_hit_rate`` (0.0–1.0; cpu/ssd
        return 0.0 until the mooncake tier collector is wired). Returns
        ``(s_cache, gpu_hit)`` so the caller logs gpu_hit without re-querying.
        """
        gpu_hit = store.get_layer_prefix_hit_rate(replica.replica_id, prompt_ids, Layer.GPU) or 0.0
        cpu_hit = store.get_layer_prefix_hit_rate(replica.replica_id, prompt_ids, Layer.CPU) or 0.0
        ssd_hit = store.get_layer_prefix_hit_rate(replica.replica_id, prompt_ids, Layer.SSD) or 0.0
        w = self.layer_weights
        s_cache = w[Layer.GPU] * gpu_hit + w[Layer.CPU] * cpu_hit + w[Layer.SSD] * ssd_hit
        return s_cache, gpu_hit

    # ── Capacity-gated token routing (CAPACITY_TOKEN_AWARE) ───────────

    def _total_token_capacity(self, store: DataStore) -> int:
        """Per-replica KV-cache token capacity = ``num_gpu_blocks × block_size``.

        ``num_gpu_blocks`` is a per-replica gauge (constant across replicas in
        practice); ``block_size`` is learned from the first KV event (defaults
        to 16 if not yet seen). Returns 0 when unavailable, in which case the
        caller falls back to least-inflight.
        """
        for node_id in store.get_metric_node_ids():
            nblk = store.get_metric(node_id, MetricKey.NUM_GPU_BLOCKS)
            if nblk and nblk > 0:
                block_size = store.get_block_size() or 16
                return int(nblk) * int(block_size)
        return 0

    def _capacity_token_scores(
        self,
        store: DataStore,
        replicas: list[ReplicaInfo],
        prompt_ids: list[int],
    ) -> list[float]:
        """Capacity-gated token routing (discrete: winner=STICKY_TOP_SCORE, rest 0).

        For each replica ``i``::

            avail[i]     = cap × (1 - kv_cache_usage_perc[i])   # free tokens (no cache)
            need[i]      = len(prompt_ids) × (1 - gpu_hit[i])    # prefill this req adds
            remaining[i] = (avail[i] - need[i]) + (cap - infight_tokens).  # free tokens after assign
            eligible[i]  = avail[i] >= cap × (1 - load_threshold)   # pure capacity gate

        Pick ``argmax(eligible, remaining)``; cold start (``kv_cache_usage_perc`` all
        ≈ 0) falls back to least-inflight;
        """
        n = len(replicas)
        cap = self._total_token_capacity(store)
        plen = len(prompt_ids) if prompt_ids else 0
        rows: list[dict] = []
        for replica in replicas:
            kv_perc = store.get_metric(replica.replica_id, MetricKey.KV_CACHE_USAGE_PERC) or 0.0
            inflight = store.get_metric(replica.replica_id, MetricKey.INFLIGHT_COUNT) or 0
            inflight_tokens = store.get_metric(replica.replica_id, MetricKey.INFLIGHT_TOKENS) or 0
            s_cache, gpu_hit = self._cache_score(store, replica, prompt_ids)
            avail = cap * (1.0 - kv_perc)
            need = plen * (1.0 - gpu_hit)
            remaining = (avail - need) + (cap - inflight_tokens)
            rows.append(
                {
                    "replica": replica,
                    "kv_perc": kv_perc,
                    "inflight": inflight,
                    "inflight_tokens": inflight_tokens,
                    "gpu_hit": gpu_hit,
                    "s_cache": s_cache,
                    "avail": avail,
                    "need": need,
                    "remaining": remaining,
                }
            )

        thresh = cap * (1.0 - self.load_threshold)
        eligible = [i for i in range(n) if rows[i]["avail"] >= thresh]
        if not eligible:
            top = max(range(n), key=lambda i: rows[i]["remaining"])
            logger.info("score(): CAPACITY_TOKEN_AWARE no eligible → max remaining")
        else:
            top = max(eligible, key=lambda i: rows[i]["remaining"])

        for i, row in enumerate(rows):
            tag = " ← WINNER" if i == top else ""
            logger.info(
                f"score(): replica={row['replica'].replica_id} kv_perc={row['kv_perc']:.3f} "
                f"gpu_hit={row['gpu_hit']:.3f} inflight={row['inflight']} "
                f"avail={row['avail']:.0f} need={row['need']:.0f} "
                f"max_num_batched_tokens={self._max_num_batched_tokens} inflight_tokens={row['inflight_tokens']:} "
                f"remaining={row['remaining']:.0f}{tag}"
            )
        winner = rows[top]["replica"].replica_id
        logger.info(
            f"score(): CAPACITY_TOKEN_AWARE winner={winner} "
            f"(kv_perc={rows[top]['kv_perc']:.3f}, remaining={rows[top]['remaining']:.0f})"
        )
        # Per-replica capacity signal for the plot (mirrors route-load in prefix-load-aware).
        cap_loads = {row["replica"].replica_id: row["remaining"] for row in rows}
        logger.info(f"route-capacity remaining={cap_loads}")
        scores = [0.0] * n
        scores[top] = STICKY_TOP_SCORE
        return scores


StrategyRegistry.register(KVCAwareStrategyConfig, KVCacheAwareStrategy)
