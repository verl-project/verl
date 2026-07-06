"""Metric specifications — canonical key names, defaults, and metadata.

This module is a **data definition layer** — it defines canonical key names,
metric metadata (defaults, types, descriptions).

Backend-specific Prometheus mappings and parsing logic live in each
backend collector, not here.
"""

from __future__ import annotations

from typing import Any

# ── Canonical key constants ──────────────────────────────────────────
# Strategies reference keys via MetricKey constants — never raw strings.


class MetricKey:
    """Canonical metric key names — backend-agnostic, strategy-layer unified."""

    KV_CACHE_USAGE_PERC: str = "kv_cache_usage_perc"
    NUM_REQUESTS_RUNNING: str = "num_requests_running"
    NUM_REQUESTS_WAITING: str = "num_requests_waiting"


# ── Metric definitions (single source of truth) ──────────────────────
# key = canonical name (matches MetricKey constant values)
# value = property dict: default / value_type / describe

METRIC_SPECS: dict[str, dict[str, Any]] = {
    MetricKey.KV_CACHE_USAGE_PERC: {
        "default": 0.0,
        "value_type": float,
        "describe": "GPU KV cache usage percentage",
    },
    MetricKey.NUM_REQUESTS_RUNNING: {
        "default": 0,
        "value_type": int,
        "describe": "Number of requests currently running",
    },
    MetricKey.NUM_REQUESTS_WAITING: {
        "default": 0,
        "value_type": int,
        "describe": "Number of requests waiting to be processed",
    },
}
