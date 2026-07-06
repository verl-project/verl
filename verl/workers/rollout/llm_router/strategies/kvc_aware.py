"""KVCache-aware runtime strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verl.workers.rollout.llm_router.config.strategy import KVCAwareStrategyConfig
from verl.workers.rollout.llm_router.logging import get_router_logger
from verl.workers.rollout.llm_router.metric_spec import MetricKey
from verl.workers.rollout.llm_router.strategies.load_score import (
    DEFAULT_LOAD_WEIGHTS,
    LOAD_FNS,
    resolve_max_num_seqs,
)
from verl.workers.rollout.llm_router.strategies.registry import StrategyRegistry

if TYPE_CHECKING:
    from verl.workers.rollout.llm_router.collectors.provider import RouteDataProvider
    from verl.workers.rollout.llm_router.strategies.base import ReplicaInfo

logger = get_router_logger("kvc-aware-strategy")

# Sticky short-circuit places the bound replica first by giving it a finite
# top score (others 0.0). Must be finite — route()'s _rank_key treats NaN/inf
# as worst — and large enough to outrank any combined score (which is bounded
# by alpha*1 + (1-alpha)*1 = 1.0).
STICKY_TOP_SCORE = 1e9


class StrategyError(Exception):
    """Strategy construction or scoring error."""


class KVCacheAwareStrategy:
    """Runtime strategy constructed from a ``KVCAwareStrategyConfig``."""

    def __init__(
        self,
        *,
        alpha: float,
        load_threshold: float,
        layer_weights: dict[str, float],
        collector_names: list[str],
        weight: float,
        load_fn: str = "normalized",
        load_weights: tuple[float, float, float] = DEFAULT_LOAD_WEIGHTS,
        max_num_seqs: int | None = None,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise StrategyError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 < load_threshold < 1:
            raise StrategyError(f"load_threshold must be in (0, 1), got {load_threshold}")
        _valid_tiers = {"gpu", "cpu", "ssd"}
        if set(layer_weights.keys()) != _valid_tiers:
            raise StrategyError(f"layer_weights keys must be {_valid_tiers}, got {set(layer_weights.keys())}")
        for tier, tier_weight in layer_weights.items():
            if tier_weight < 0:
                raise StrategyError(f"layer_weights[{tier}] must be >= 0, got {tier_weight}")
        weights_sum = sum(layer_weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise StrategyError(f"layer_weights values must sum to 1.0, got {weights_sum}")
        if load_fn not in LOAD_FNS:
            raise StrategyError(f"load_fn must be one of {list(LOAD_FNS)}, got '{load_fn}'")
        if len(load_weights) != 3:
            raise StrategyError(f"load_weights must have 3 elements (a,b,c), got len={len(load_weights)}")
        if any(w < 0 for w in load_weights):
            raise StrategyError(f"load_weights must be >= 0, got {load_weights}")
        if abs(sum(load_weights) - 1.0) > 1e-6:
            raise StrategyError(f"load_weights must sum to 1.0, got {sum(load_weights)}")

        self.alpha = float(alpha)
        self.load_threshold = float(load_threshold)
        self.layer_weights = dict(layer_weights)
        self.collector_names = collector_names
        self.weight = weight
        self.load_fn = load_fn
        self.load_weights = tuple(load_weights)
        self._load_fn = LOAD_FNS[load_fn]
        # max_num_seqs drives the normalized load formula's running term.
        # None → resolve from the MAX_NUM_SEQS env var (default 64).
        self._max_num_seqs = int(max_num_seqs) if max_num_seqs is not None else resolve_max_num_seqs()
        logger.info(
            f"KVCacheAwareStrategy created: alpha={self.alpha:.2f}, "
            f"load_threshold={self.load_threshold:.2f}, layer_weights={self.layer_weights}, "
            f"load_fn='{self.load_fn}', load_weights={self.load_weights}, max_num_seqs={self._max_num_seqs}",
        )

    @classmethod
    def from_config(cls, cfg: KVCAwareStrategyConfig) -> KVCacheAwareStrategy:
        """Construct a strategy instance carrying its parsed config fields.

        Load-function selection (``load_fn`` / ``load_weights``) is code-level —
        not exposed via YAML — so ``from_config`` uses code defaults. The
        ``load_threshold`` field (semantics: overload when ``load > threshold``)
        comes from the config.
        """
        return cls(
            alpha=cfg.alpha,
            load_threshold=cfg.load_threshold,
            layer_weights=cfg.layer_weights,
            collector_names=cfg.collector_names,
            weight=cfg.weight,
        )

    # ── Load scoring ──

    def _compute_load(
        self,
        kv_usage: float,
        running: int | float,
        waiting: int | float,
    ) -> float:
        """Compute ``load ∈ [0,1]`` (bigger = more loaded) via the selected fn."""
        return self._load_fn(
            kv_usage,
            running,
            waiting,
            max_num_seqs=self._max_num_seqs,
            weights=self.load_weights,
        )

    def is_overloaded(
        self,
        provider: RouteDataProvider,
        replica: ReplicaInfo,
    ) -> bool:
        """Return True if ``replica`` is overloaded (``load > load_threshold``).

        Used only by the sticky short-circuit to decide whether to send a
        returning session back to its bound replica. Combined scoring never
        consults overload.
        """
        m = provider.get_metrics(replica.replica_id)
        kv_usage = m.get(MetricKey.KV_CACHE_USAGE_PERC, 0.0)
        running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
        waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
        return self._compute_load(kv_usage, running, waiting) > self.load_threshold

    def _sticky_shortcut(
        self,
        provider: RouteDataProvider,
        replicas: list[ReplicaInfo],
        request_id: str | None,
        sticky_table: Any,
    ) -> list[float] | None:
        """Return a pre-built score list if a sticky replica should win, else None.

        Sticky replica wins when: ``request_id``/``sticky_table`` are provided,
        the bound replica is present in ``replicas``, and it is NOT overloaded.
        On win, returns a list with ``STICKY_TOP_SCORE`` at the bound replica's
        index and ``0.0`` elsewhere. On miss / overload / absence, returns
        ``None`` so the caller falls through to combined scoring.
        """
        if not request_id or sticky_table is None:
            return None
        sticky_id = sticky_table.get(request_id)
        if sticky_id is None:
            return None
        for idx, replica in enumerate(replicas):
            if replica.replica_id == sticky_id:
                m = provider.get_metrics(replica.replica_id)
                kv_usage = m.get(MetricKey.KV_CACHE_USAGE_PERC, 0.0)
                running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
                waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
                load = self._compute_load(kv_usage, running, waiting)
                metrics_str = (
                    f"kv={kv_usage:.3f} running={running} waiting={waiting} "
                    f"→ load={load:.4f} s_load={1.0 - load:.4f} (threshold={self.load_threshold:.2f})"
                )
                if load > self.load_threshold:
                    logger.info(
                        f"score(): STICKY replica={sticky_id} OVERLOADED [{metrics_str}] → fallback to COMBINED scoring"
                    )
                    return None
                logger.info(
                    f"score(): STICKY replica={sticky_id} HIT (not overloaded) [{metrics_str}] "
                    f"→ short-circuit (top score)"
                )
                scores = [0.0] * len(replicas)
                scores[idx] = STICKY_TOP_SCORE
                return scores
        # Bound replica no longer in pool — let the Balancer invalidate it.
        logger.info(
            f"score(): sticky replica={sticky_id} not in pool, fallback to combined scoring",
        )
        return None

    def score(
        self,
        prompt_ids: list[int] | None,
        provider: RouteDataProvider,
        replicas: list[ReplicaInfo],
        request_id: str | None = None,
        sticky_table: Any = None,
    ) -> list[float]:
        """Score each replica. Larger is better.

        Combined score (one pass — no fast/slow branching, no cache zeroing):
            S = α·S_cache + (1-α)·S_load
            S_cache = w_gpu·gpu_hit + w_cpu·cpu_hit + w_ssd·ssd_hit   (weights sum to 1)
            S_load  = 1 - load                                         (bigger = less loaded)
            load    = selected load_fn(kv, running, waiting, max_num_seqs, weights)

        Every replica — overloaded or not — gets the full formula. Overload is
        consulted only by the sticky short-circuit (``is_overloaded``).

        Sticky short-circuit: when ``request_id`` and ``sticky_table`` are
        provided and the bound replica is present and NOT overloaded, returns a
        pre-built score list placing that replica first (sticky replica gets
        ``STICKY_TOP_SCORE``, others ``0.0``), skipping combined scoring.
        """
        if not isinstance(replicas, list):
            raise StrategyError(f"replicas must be a list, got {type(replicas).__name__}")
        if not replicas:
            return []

        # Sticky short-circuit: bound, non-overloaded replica wins outright.
        shortcut = self._sticky_shortcut(provider, replicas, request_id, sticky_table)
        if shortcut is not None:
            return shortcut

        effective_prompt_ids = prompt_ids or []

        # GPU prefix hit is the same prompt for every replica — query once.
        # get_gpu_prefix_hit_rate returns {replica_id: 0-100}; _cache_score
        # scales each replica's value to 0-1.
        gpu_hit_pct = provider.get_gpu_prefix_hit_rate(effective_prompt_ids)

        result = []
        for replica in replicas:
            m = provider.get_metrics(replica.replica_id)
            kv_usage = m.get(MetricKey.KV_CACHE_USAGE_PERC, 0.0)
            running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
            waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
            load = self._compute_load(kv_usage, running, waiting)
            s_load = 1.0 - load
            s_cache = self._cache_score(provider, replica, effective_prompt_ids, gpu_hit_pct)
            score = self.alpha * s_cache + (1 - self.alpha) * s_load
            result.append(score)
            gpu_hit = gpu_hit_pct.get(replica.replica_id, 0) / 100.0
            logger.info(
                f"score(): replica={replica.replica_id} kv={kv_usage:.3f} running={running} waiting={waiting} "
                f"→ load={load:.4f} s_load={s_load:.4f} | gpu_hit={gpu_hit:.2f} s_cache={s_cache:.4f} "
                f"({self.alpha:.2f}·cache + {1 - self.alpha:.2f}·load) → score={score:.4f}"
            )
        scores_str = ", ".join(f"{r.replica_id}={result[i]:.4f}" for i, r in enumerate(replicas))
        logger.info(f"score(): COMBINED scores: {scores_str}")
        return result

    def _cache_score(
        self,
        provider: RouteDataProvider,
        replica: ReplicaInfo,
        prompt_ids: list[int],
        gpu_hit_pct: dict[str, float],
    ) -> float:
        """Three-layer weighted prefix-cache hit score ∈ [0, 1].

            S_cache = w_gpu·gpu_hit + w_cpu·cpu_hit + w_ssd·ssd_hit

        ``gpu_hit`` comes from ``get_gpu_prefix_hit_rate`` (0-100, scaled to
        0-1); ``cpu_hit``/``ssd_hit`` come from ``get_tier_prefix_hit_rate``
        (``None`` → 0.0, e.g. when the mooncake tier collector is not yet
        implemented). Weights come from ``self.layer_weights`` and are
        validated to sum to 1.0 at construction, so the result is already in
        [0, 1] with no extra normalization.
        """
        gpu_hit = gpu_hit_pct.get(replica.replica_id, 0) / 100.0
        cpu_hit = provider.get_tier_prefix_hit_rate(replica.replica_id, prompt_ids, "cpu") or 0.0
        ssd_hit = provider.get_tier_prefix_hit_rate(replica.replica_id, prompt_ids, "ssd") or 0.0
        w = self.layer_weights
        return w["gpu"] * gpu_hit + w["cpu"] * cpu_hit + w["ssd"] * ssd_hit


# Auto-register: config dataclass type → runtime strategy class.
StrategyRegistry.register(KVCAwareStrategyConfig, KVCacheAwareStrategy)
