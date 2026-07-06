"""VLLMMetricsDecoder — vLLM Prometheus metrics decoder.

Parses Prometheus exposition-format text and writes results
to MetricsStore.
"""

from __future__ import annotations

import logging
from typing import Any

from verl.workers.rollout.llm_router.collectors.decoder.base import Decoder
from verl.workers.rollout.llm_router.metric_spec import METRIC_SPECS, MetricKey
from verl.workers.rollout.llm_router.store.metrics_store import MetricsStore

logger = logging.getLogger(__name__)


class VLLMMetricsDecoder(Decoder):
    """vLLM Prometheus metrics decoder — parses HTTP response text
    and writes results to MetricsStore.

    vLLM Prometheus raw name → canonical key mapping:
        ``vllm:kv_cache_usage_perc``  → ``KV_CACHE_USAGE_PERC``
        ``vllm:num_requests_running`` → ``NUM_REQUESTS_RUNNING``
        ``vllm:num_requests_waiting`` → ``NUM_REQUESTS_WAITING``
    """

    store_cls = MetricsStore

    _PROMETHEUS_MAP: dict[str, str] = {
        "vllm:kv_cache_usage_perc": MetricKey.KV_CACHE_USAGE_PERC,
        "vllm:num_requests_running": MetricKey.NUM_REQUESTS_RUNNING,
        "vllm:num_requests_waiting": MetricKey.NUM_REQUESTS_WAITING,
    }

    def __init__(self) -> None:
        self._store = self.store_cls.default()

    def decode(self, raw_data: bytes | str, node_id: str) -> None:
        """Parse Prometheus text and write results to store.

        Args:
            raw_data: HTTP response text (Prometheus exposition format).
            node_id: Source replica identifier.
        """
        # HTTP delivers str; ignore bytes data
        if isinstance(raw_data, bytes):
            logger.debug("VLLMMetricsDecoder received bytes data, expected str — skipping")
            return

        result: dict[str, Any] = {}
        for line in raw_data.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            try:
                raw_name = line.split("{")[0] if "{" in line else line.split()[0]
                value = float(line.split()[-1])
            except (ValueError, IndexError):
                continue
            canonical = self._PROMETHEUS_MAP.get(raw_name)
            if canonical:
                value_type = METRIC_SPECS[canonical].get("value_type", float)
                result[canonical] = value_type(value)

        self._store.refresh({node_id: result})
