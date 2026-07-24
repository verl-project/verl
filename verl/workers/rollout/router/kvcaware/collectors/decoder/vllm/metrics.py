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

"""VLLMMetricsDecoder — vLLM Prometheus metrics decoder.

Parses Prometheus exposition-format text and returns structured metrics.
Store writes are handled by Collector via DataStore.
"""

from __future__ import annotations

from typing import Any

from ....collectors.decoder import Decoder, MetricsUpdate
from ....logging import get_router_logger
from ....types import METRIC_SPECS, MetricKey

logger = get_router_logger("vllm-metrics")


class VLLMMetricsDecoder(Decoder):
    """vLLM Prometheus metrics decoder — parses HTTP response text.

    Returns structured metrics.

    vLLM Prometheus raw name → canonical key mapping:
        ``vllm:kv_cache_usage_perc``  → ``KV_CACHE_USAGE_PERC``
        ``vllm:num_requests_running`` → ``NUM_REQUESTS_RUNNING``
        ``vllm:num_requests_waiting`` → ``NUM_REQUESTS_WAITING``
    """

    _PROMETHEUS_MAP: dict[str, str] = {
        "vllm:kv_cache_usage_perc": MetricKey.KV_CACHE_USAGE_PERC,
        "vllm:num_requests_running": MetricKey.NUM_REQUESTS_RUNNING,
        "vllm:num_requests_waiting": MetricKey.NUM_REQUESTS_WAITING,
        # vLLM 0.21 Prometheus counters carry a ``_total`` suffix (confirmed by
        # scraping /metrics); the pre-Task-2 names without ``_total`` never matched.
        "vllm:prefix_cache_queries_total": MetricKey.PREFIX_CACHE_QUERIES,
        "vllm:prefix_cache_hits_total": MetricKey.PREFIX_CACHE_HITS,
        # Evidence metrics: TTFT/TPOT histograms + token/external counters.
        # All cumulative — the periodic log emits windowed deltas (rates/averages).
        "vllm:time_to_first_token_seconds_sum": MetricKey.TTFT_SECONDS_SUM,
        "vllm:time_to_first_token_seconds_count": MetricKey.TTFT_COUNT,
        # Queue wait (TTFT includes it; prefill_time = TTFT - queue is the real
        # prefill cost that prefix-sharing reduces).
        "vllm:request_queue_time_seconds_sum": MetricKey.QUEUE_TIME_SECONDS_SUM,
        "vllm:request_queue_time_seconds_count": MetricKey.QUEUE_TIME_COUNT,
        # NOTE: TPOT histogram is ``request_time_per_output_token_seconds`` in
        # vLLM 0.21 (NOT ``time_per_output_token_seconds``).
        "vllm:request_time_per_output_token_seconds_sum": MetricKey.TPOT_SECONDS_SUM,
        "vllm:request_time_per_output_token_seconds_count": MetricKey.TPOT_COUNT,
        "vllm:generation_tokens_total": MetricKey.GENERATION_TOKENS,
        "vllm:external_prefix_cache_hits_total": MetricKey.EXTERNAL_PREFIX_CACHE_HITS,
        # Analytic FLOPs counter (gated by vLLM --enable-mfu-metrics); for MFU.
        "vllm:estimated_flops_per_gpu_total": MetricKey.ESTIMATED_FLOPS_PER_GPU,
        # PROMPT_TOKENS / PROMPT_TOKENS_CACHED are NOT here — they come from the
        # labeled ``prompt_tokens_by_source_total{source=...}`` metric, dispatched
        # label-aware in ``_resolve_canonical`` (cache_hit vs local_compute).
        # cache_config_info is also NOT here — it's an info gauge whose value is
        # 1.0; the real ``num_gpu_blocks`` lives in a label and is extracted
        # separately in ``decode`` (label-as-value).
    }

    def decode(self, raw_data: bytes | str, node_id: str) -> MetricsUpdate | None:
        """Parse Prometheus text and return structured metrics.

        Args:
            raw_data: HTTP response text (Prometheus exposition format).
            node_id: Source endpoint identifier.

        Returns:
            MetricsUpdate with parsed metrics, or None if decode failed.
        """
        # HTTP delivers str; ignore bytes data
        if isinstance(raw_data, bytes):
            logger.debug("VLLMMetricsDecoder received bytes data, expected str — skipping")
            return None

        result: dict[str, Any] = {}
        for line in raw_data.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            # Label-aware parse: Prometheus exposition is
            #   ``metric_name{label="v",...}<ws>value``
            # We need labels for ``prompt_tokens_by_source_total{source=...}`` to
            # split cache-hit vs computed prompt tokens, and for
            # ``cache_config_info`` to read ``num_gpu_blocks``.
            try:
                if "{" in line:
                    name_part, rest = line.split("{", 1)
                    labels_str, _, value_part = rest.partition("}")
                    raw_name = name_part.strip()
                    labels: dict[str, str] = {}
                    for kv in labels_str.split(","):
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            labels[k.strip()] = v.strip().strip('"')
                else:
                    # No-label line: ``name<ws>value`` — split once, reuse parts.
                    parts = line.split()
                    raw_name = parts[0]
                    labels = {}
                    value_part = parts[-1] if len(parts) > 1 else ""
            except (ValueError, IndexError):
                logger.warning(f"decode failed: {line}")
                continue

            # cache_config_info is an info gauge: its value is 1.0, the real
            # data lives in the ``num_gpu_blocks`` label.
            if raw_name == "vllm:cache_config_info":
                n = labels.get("num_gpu_blocks")
                if n is not None:
                    try:
                        result[MetricKey.NUM_GPU_BLOCKS] = int(float(n))
                    except ValueError:
                        pass
                continue
            try:
                value = float(value_part)
            except ValueError:
                continue
            canonical = self._resolve_canonical(raw_name, labels)
            if canonical:
                value_type = METRIC_SPECS[canonical].get("value_type", float)
                result[canonical] = value_type(value)

        logger.debug(f"vllm-metrics replica={node_id} polled: {result}")
        return MetricsUpdate(node_id=node_id, metrics=result)

    @staticmethod
    def _resolve_canonical(raw_name: str, labels: dict[str, str]) -> str | None:
        """Map a scraped metric name (+labels) to a canonical key.

        Most metrics are label-free lookups in ``_PROMETHEUS_MAP``. The
        ``prompt_tokens_by_source_total`` metric carries a ``source`` label that
        splits prompt tokens into cache-hit vs locally-computed — the cleanest
        evidence signal for prefix reuse — so dispatch on the label.
        """
        if raw_name == "vllm:prompt_tokens_by_source_total":
            source = labels.get("source")
            if source == "local_cache_hit":
                return MetricKey.PROMPT_TOKENS_CACHED
            if source == "local_compute":
                return MetricKey.PROMPT_TOKENS
            return None
        return VLLMMetricsDecoder._PROMETHEUS_MAP.get(raw_name)
