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
HTTP /metrics scraping helpers for least_kv_cache routing.

Parsing is intentionally minimal: skip # comment lines, keep lines whose metric
identifier matches metric_name, then take the last numeric literal on the line
(as the Prometheus sample value is typically last).
"""

from __future__ import annotations

import re
import urllib.request
from typing import Optional

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def build_metrics_url(server_address: str, metrics_path: str = "/metrics") -> str:
    """Build http://host:port/metrics from a rollout replica address string.

    server_address matches RolloutReplica._server_address (host:port or [ipv6]:port).
    """
    if not metrics_path.startswith("/"):
        metrics_path = "/" + metrics_path
    return f"http://{server_address}{metrics_path}"


def fetch_prometheus_text(url: str, timeout_s: float = 2.0) -> str:
    """HTTP GET returning Prometheus text exposition body."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _metric_name_prefix_match(line: str, metric_name: str) -> bool:
    """True if line is a sample for metric_name (name before {{ or whitespace)."""
    if not line.startswith(metric_name):
        return False
    if len(line) == len(metric_name):
        return True
    return line[len(metric_name)] in "{ \t"


def _last_float_on_line(line: str) -> Optional[float]:
    matches = _FLOAT_RE.findall(line)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def parse_prometheus_metric_value(text: str, metric_name: Optional[str] = None) -> Optional[float]:
    """Max of last numeric literals on non-comment lines whose metric identifier is metric_name.

    Comment / metadata lines (leading # after strip) are ignored.
    """
    if not metric_name:
        return None

    if text.startswith("\ufeff"):
        text = text[1:]

    values: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not _metric_name_prefix_match(line, metric_name):
            continue
        v = _last_float_on_line(line)
        if v is not None:
            values.append(v)
    if not values:
        return None
    return max(values)


def collect_matching_metric_sample_lines(text: str, metric_name: str, max_lines: int = 50) -> list[str]:
    """Return stripped sample lines whose metric identifier equals metric_name (debug helper)."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _metric_name_prefix_match(line, metric_name):
            out.append(line)
            if len(out) >= max_lines:
                break
    return out
