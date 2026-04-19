# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Lightweight Prometheus text exposition parsing for per-replica HTTP /metrics scraping.

Used by GlobalRequestLoadBalancer when ``load_balance_strategy`` is ``least_kv_cache``.
"""

from __future__ import annotations

import urllib.request
from typing import Optional


def build_metrics_url(server_address: str, metrics_path: str = "/metrics") -> str:
    """Build ``http://host:port/metrics`` from a rollout replica address string.

    ``server_address`` matches ``RolloutReplica._server_address`` (``host:port`` or ``[ipv6]:port``).
    """
    if not metrics_path.startswith("/"):
        metrics_path = "/" + metrics_path
    return f"http://{server_address}{metrics_path}"


def fetch_prometheus_text(url: str, timeout_s: float = 2.0) -> str:
    """HTTP GET returning Prometheus text exposition body."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _split_metric_line(line: str) -> tuple[str, str] | None:
    """Return (metric_with_labels, value_token) or None if not a sample line."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.rsplit(None, 1)
    if len(parts) != 2:
        return None
    metric_part, value_token = parts
    try:
        float(value_token)
    except ValueError:
        return None
    return metric_part, value_token


def _metric_identifier(metric_with_labels: str) -> str:
    """Metric name before labels, e.g. ``foo`` from ``foo{a="b"}``."""
    brace = metric_with_labels.find("{")
    if brace == -1:
        return metric_with_labels
    return metric_with_labels[:brace]


def parse_prometheus_metric_value(text: str, metric_name: Optional[str] = None) -> Optional[float]:
    """Aggregate matching samples for ``metric_name`` (identifier before ``{`` must equal ``metric_name``).

    A single scrape can expose **multiple lines** with the same metric name and different label sets, e.g.::

        vllm:kv_cache_usage_perc{model_name="a"} 0.2
        vllm:kv_cache_usage_perc{model_name="b"} 0.6

    For load-aware routing we need one scalar per replica; we take the **maximum** among all matching
    samples so that replicas with any hot partition rank as more loaded (conservative for
    ``least_kv_cache``).

    Returns ``None`` if ``metric_name`` is unset or no matching sample exists.
    """
    if not metric_name:
        return None

    values: list[float] = []
    for line in text.splitlines():
        parsed = _split_metric_line(line)
        if parsed is None:
            continue
        metric_part, value_token = parsed
        ident = _metric_identifier(metric_part)
        if ident != metric_name:
            continue
        try:
            values.append(float(value_token))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)
