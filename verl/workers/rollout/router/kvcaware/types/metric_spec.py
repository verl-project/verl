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
    # Per-replica total GPU KV blocks — from the ``num_gpu_blocks`` label of
    # vLLM's ``cache_config_info`` gauge (the value itself is 1.0; the real
    # number lives in the label). Used to turn retained-block counts into an
    # occupancy ratio that, unlike kv_cache_usage_perc, reflects the free pool.
    NUM_GPU_BLOCKS: str = "num_gpu_blocks"
    PREFIX_CACHE_QUERIES: str = "prefix_cache_queries"
    PREFIX_CACHE_HITS: str = "prefix_cache_hits"
    # ── Evidence metrics — cumulative counters/histograms polled from vLLM
    # /metrics. Use delta between snapshots for rates.
    TTFT_SECONDS_SUM: str = "ttft_seconds_sum"
    TTFT_COUNT: str = "ttft_count"
    # Queue time (TTFT includes queue wait; prefill_time = TTFT - queue is what
    # prefix-sharing actually reduces). Histogram, like TTFT.
    QUEUE_TIME_SECONDS_SUM: str = "queue_time_seconds_sum"
    QUEUE_TIME_COUNT: str = "queue_time_count"
    TPOT_SECONDS_SUM: str = "tpot_seconds_sum"
    TPOT_COUNT: str = "tpot_count"
    # Prefill tokens actually computed (cache MISS) — the "recompute length".
    PROMPT_TOKENS: str = "prompt_tokens"
    # Prompt tokens that hit cache — the "prefix hit length" in tokens.
    PROMPT_TOKENS_CACHED: str = "prompt_tokens_cached"
    # Decode tokens generated.
    GENERATION_TOKENS: str = "generation_tokens"
    # External (mc connector) prefix-cache hits — cross-replica KV reuse.
    EXTERNAL_PREFIX_CACHE_HITS: str = "external_prefix_cache_hits"
    # Estimated FLOPs per GPU (vLLM analytic counter; MFU = rate(flops) / peak_flops).
    ESTIMATED_FLOPS_PER_GPU: str = "estimated_flops_per_gpu"
    # In-flight request count (acquire +1 / release -1, mirrors verl least-inflight).
    INFLIGHT_COUNT: str = "inflight_count"
    # In-flight prompt tokens (acquire +prompt_len / release -prompt_len). The
    # token-weighted sibling of INFLIGHT_COUNT: a single 30k-token request and a
    # 100-token one both count as 1 inflight, but load the KV cache very
    # differently — this gauge captures that. acquire/release share the same
    # request's prompt_len in one generate() scope, so the gauge is symmetric.
    INFLIGHT_TOKENS: str = "inflight_tokens"
    # Cumulative dispatched request count (acquire +1) — the running inflight
    # gauge's monotonic sibling. Per-replica; sum across replicas for the global
    # dispatched volume. Paired with COMPLETED_COUNT for dispatch/complete rates.
    DISPATCHED_COUNT: str = "dispatched_count"
    # Cumulative completed request count (release +1). Per-replica; paired with
    # DISPATCHED_COUNT — completed/dispatched ratio is the realized-throughput share.
    COMPLETED_COUNT: str = "completed_count"
    # Cumulative sum of per-request turn at dispatch, per replica. Each dispatch's
    # turn (Nth time that request_id was dispatched globally) is added to the
    # receiving replica's TURN_SUM. Windowed avg turn = Δturn_sum / Δdispatched.
    TURN_SUM: str = "turn_sum"
    # Cumulative sum of dispatched prompt lengths (len(prompt_ids)), per replica.
    # Each dispatch adds the request's input prompt length to the receiving
    # replica's PROMPT_LEN_SUM. Windowed avg prompt len = Δsum / Δdispatched — the
    # request-size signal the router sees at dispatch time (pre-engine).
    PROMPT_LEN_SUM: str = "prompt_len_sum"


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
    MetricKey.NUM_GPU_BLOCKS: {
        "default": 0,
        "value_type": int,
        "describe": "Per-replica total GPU KV blocks (cache_config_info num_gpu_blocks label)",
    },
    MetricKey.PREFIX_CACHE_QUERIES: {
        "default": 0,
        "value_type": int,
        "describe": "Prefix cache query count (vLLM counter; cumul. — use delta for rate)",
    },
    MetricKey.PREFIX_CACHE_HITS: {
        "default": 0,
        "value_type": int,
        "describe": "Prefix cache hit count (vLLM counter; cumul. — use delta for rate)",
    },
    # ── Evidence metrics — all cumulative; consume deltas for rates/averages. ──
    MetricKey.TTFT_SECONDS_SUM: {
        "default": 0.0,
        "value_type": float,
        "describe": "Cumul. sum of time-to-first-token seconds (histogram _sum)",
    },
    MetricKey.TTFT_COUNT: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. count of TTFT samples (histogram _count) = #prefills",
    },
    MetricKey.QUEUE_TIME_SECONDS_SUM: {
        "default": 0.0,
        "value_type": float,
        "describe": "Cumul. sum of request queue-wait seconds (TTFT incl. queue; prefill=TTFT-queue)",
    },
    MetricKey.QUEUE_TIME_COUNT: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. count of queue-time samples (histogram _count)",
    },
    MetricKey.TPOT_SECONDS_SUM: {
        "default": 0.0,
        "value_type": float,
        "describe": "Cumul. sum of time-per-output-token seconds (histogram _sum)",
    },
    MetricKey.TPOT_COUNT: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. count of TPOT samples (histogram _count)",
    },
    MetricKey.PROMPT_TOKENS: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. prefill tokens computed (cache miss) — recompute length",
    },
    MetricKey.PROMPT_TOKENS_CACHED: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. prompt tokens served from cache — prefix-hit length (tokens)",
    },
    MetricKey.GENERATION_TOKENS: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. decode tokens generated",
    },
    MetricKey.EXTERNAL_PREFIX_CACHE_HITS: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. external (mc connector) prefix-cache hits — cross-replica KV reuse",
    },
    MetricKey.ESTIMATED_FLOPS_PER_GPU: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. estimated FLOPs per GPU (vLLM analytic; MFU = rate / peak_flops)",
    },
    MetricKey.INFLIGHT_COUNT: {
        "default": 0,
        "value_type": int,
        "describe": "In-flight request count (acquire +1 / release -1, verl least-inflight signal)",
    },
    MetricKey.INFLIGHT_TOKENS: {
        "default": 0,
        "value_type": int,
        "describe": "In-flight prompt tokens (acquire +prompt_len / release -prompt_len) — token-weighted load",
    },
    MetricKey.DISPATCHED_COUNT: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. dispatched requests per replica (acquire +1) — pair with COMPLETED_COUNT for rates",
    },
    MetricKey.COMPLETED_COUNT: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. completed requests per replica (release +1) — realized-throughput share",
    },
    MetricKey.TURN_SUM: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. sum of per-request turn at dispatch, per replica (avg turn = Δturn_sum/Δdispatched)",
    },
    MetricKey.PROMPT_LEN_SUM: {
        "default": 0,
        "value_type": int,
        "describe": "Cumul. sum of dispatched prompt lengths, per replica (avg len = Δsum/Δdispatched)",
    },
}
