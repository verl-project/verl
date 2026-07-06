"""Selectable load-score functions for KVCacheAwareStrategy.

Every function returns ``load ∈ [0, 1]`` with the convention **bigger = more
loaded** (a saturated replica scores 1.0, an idle one 0.0). The strategy
converts to its internal ``s_load`` via ``s_load = 1 - load`` so the combined
score ``α·S_cache + (1-α)·S_load`` keeps "bigger = preferred".

Available functions (select by name via ``LOAD_FNS`` / ``get_load_fn``):
    - ``"normalized"`` (default): convex combination of three normalized signals
        running_usage = min(1, running / max_num_seqs)
        waiting_usage = min(1, waiting / max_num_seqs)
        load = a·kv_usage + b·running_usage + c·waiting_usage     (a+b+c=1)
      Default weights (a, b, c) = (0.4, 0.3, 0.3); ``max_num_seqs`` from env
      ``MAX_NUM_SEQS`` (default 64).
    - ``"kv_over_pressure"`` (legacy): ``load = 1 - (1-kv)/(1+running+waiting)``
      — the inverse of the pre-Phase-7 ``s_load``. Kept as a selectable
      fallback / for A-B comparison; ignores weights and max_num_seqs.

Selection is code-level (constructor ``load_fn`` name), not a YAML knob.
"""

from __future__ import annotations

import os
from typing import Callable

# Default (a, b, c) weights for the normalized formula — kv / running / waiting.
DEFAULT_LOAD_WEIGHTS: tuple[float, float, float] = (0.4, 0.3, 0.3)

# Default scheduler capacity when MAX_NUM_SEQS is unset. Matches the project's
# vLLM 24GB default (scripts/infer_multi.sh).
DEFAULT_MAX_NUM_SEQS: int = 64


def load_normalized(
    kv_usage: float,
    running: int | float,
    waiting: int | float,
    *,
    max_num_seqs: int,
    weights: tuple[float, float, float] = DEFAULT_LOAD_WEIGHTS,
) -> float:
    """Convex combination of KV fullness, scheduler occupancy, and backlog share.

    Each term is normalized to [0, 1] and the weights sum to 1, so the result
    is in [0, 1] (bigger = more loaded).
    """
    a, b, c = weights
    # running_usage: fraction of scheduler capacity held by running sequences.
    # Clamp to 1.0 (running can transiently exceed max_num_seqs); max_num_seqs=0
    # degrades to fully occupied so we never divide by zero.
    if max_num_seqs > 0:
        running_usage = min(1.0, float(running) / float(max_num_seqs))
    else:
        running_usage = 1.0
    # waiting_usage: fraction of scheduler capacity backlogged, mirroring
    # running_usage's normalization (clamp to 1.0 — waiting can transiently
    # exceed max_num_seqs under KV pressure). max_num_seqs=0 → fully occupied,
    # so we never divide by zero.
    if max_num_seqs > 0:
        waiting_usage = min(1.0, float(waiting) / float(max_num_seqs))
    else:
        waiting_usage = 1.0
    return a * float(kv_usage) + b * running_usage + c * waiting_usage


def load_kv_over_pressure(
    kv_usage: float,
    running: int | float,
    waiting: int | float,
    *,
    max_num_seqs: int,
    weights: tuple[float, float, float] = DEFAULT_LOAD_WEIGHTS,
) -> float:
    """Legacy load formula — inverse of the pre-Phase-7 ``s_load``.

    ``s_load_old = (1 - kv_usage) / (1 + running + waiting)`` (bigger = less
    loaded), so ``load = 1 - s_load_old`` (bigger = more loaded). ``max_num_seqs``
    and ``weights`` are accepted for signature uniformity but unused.
    """
    s_load_old = (1.0 - float(kv_usage)) / (1.0 + float(running) + float(waiting))
    return 1.0 - s_load_old


# Registry of selectable load functions. Add a name here to expose a new one.
LOAD_FNS: dict[str, Callable[..., float]] = {
    "normalized": load_normalized,
    "kv_over_pressure": load_kv_over_pressure,
}


def get_load_fn(name: str) -> Callable[..., float]:
    """Look up a load function by name; raise ``KeyError`` if unknown."""
    return LOAD_FNS[name]


def resolve_max_num_seqs() -> int:
    """Read scheduler capacity from ``MAX_NUM_SEQS`` env var.

    Returns ``DEFAULT_MAX_NUM_SEQS`` (64) when unset or non-parseable, so the
    router degrades gracefully rather than crashing on a misconfigured env.
    """
    raw = os.environ.get("MAX_NUM_SEQS")
    if raw is None:
        return DEFAULT_MAX_NUM_SEQS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_NUM_SEQS
    return value if value > 0 else DEFAULT_MAX_NUM_SEQS
