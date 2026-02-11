# Copyright 2025 Meituan Ltd. and/or its affiliates
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


import logging
import os
import time

import ray
import requests
import yaml

from verl.workers.config.rollout import PrometheusConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def update_prometheus_config(config: PrometheusConfig, server_addresses: list[str], rollout_name: str | None = None):
    """
    Update Prometheus configuration file with server addresses and reload on first node.

    server_addresses: vllm or sglang server addresses

    rollout_name: name of the rollout backend (e.g., "vllm", "sglang")
    """

    if not server_addresses:
        logger.warning("No server addresses available to update Prometheus config")
        return

    try:
        # Get Prometheus config file path from environment or use default
        prometheus_config_json = {
            "global": {"scrape_interval": "10s", "evaluation_interval": "10s"},
            "scrape_configs": [
                {
                    "job_name": "ray",
                    "file_sd_configs": [{"files": ["/tmp/ray/prom_metrics_service_discovery.json"]}],
                },
                {"job_name": "rollout", "static_configs": [{"targets": server_addresses}]},
            ],
        }

        # Write configuration file to all nodes
        @ray.remote(num_cpus=0)
        def write_config_file(config_data, config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            return True

        # Reload Prometheus on all nodes. Only master node should succeed, skip errors on other nodes.
        @ray.remote(num_cpus=0)
        def reload_prometheus(port):
            import socket
            import subprocess

            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)

            reload_url = f"http://{ip_address}:{port}/-/reload"

            try:
                subprocess.run(["curl", "-X", "POST", reload_url], capture_output=True, text=True, timeout=10)
                print(f"Reloading Prometheus on node: {reload_url}")
            except Exception:
                # Skip errors on non-master nodes
                pass

        # Get all available nodes and schedule tasks on each node
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node["Alive"]]

        # Write config files on all nodes
        write_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = write_config_file.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(prometheus_config_json, config.file)
            write_tasks.append(task)

        ray.get(write_tasks)

        server_type = rollout_name.upper() if rollout_name else "rollout"
        print(f"Updated Prometheus configuration at {config.file} with {len(server_addresses)} {server_type} servers")

        # Reload Prometheus on all nodes
        reload_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = reload_prometheus.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(config.port)
            reload_tasks.append(task)

        ray.get(reload_tasks)

    except Exception as e:
        logger.error(f"Failed to update Prometheus configuration: {e}")


class PrometheusClient:
    """Client for querying Prometheus metrics during training.

    This client queries Prometheus running on the Ray head node to fetch
    infrastructure metrics (GPU cache usage, throughput, etc.) and makes
    them available for experiment tracking.

    Features:
    - Automatic head node discovery via Ray
    - Retry logic with exponential backoff
    - Per-metric error handling (one failure doesn't affect others)
    - Result caching to reduce query frequency

    Attributes:
        host: Prometheus host (Ray head node IP)
        port: Prometheus port (default 9090)
        metrics_to_track: List of Prometheus metric names or queries
        timeout: HTTP request timeout in seconds
        max_attempts: Maximum retry attempts per metric
        retry_delay: Base delay between retries
        cache_duration: How long to cache results (seconds)
    """

    DEFAULT_TIMEOUT = 5.0
    DEFAULT_MAX_ATTEMPTS = 2
    DEFAULT_RETRY_DELAY = 0.5
    DEFAULT_CACHE_DURATION = 10.0

    def __init__(
        self,
        prometheus_config: PrometheusConfig,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        cache_duration: float = DEFAULT_CACHE_DURATION,
    ):
        """Initialize Prometheus client.

        Args:
            prometheus_config: PrometheusConfig object from rollout config
            timeout: HTTP timeout for queries
            max_attempts: Number of retry attempts
            retry_delay: Base delay for exponential backoff
            cache_duration: Cache results for this many seconds
        """
        self.port = prometheus_config.port
        self.metrics_to_track = prometheus_config.metrics_to_track or []
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.cache_duration = cache_duration

        self.host = self._get_ray_head_node()
        self.base_url = f"http://{self.host}:{self.port}"

        self._cache = {}
        self._cache_timestamps = {}

        if self.metrics_to_track:
            logger.info(f"PrometheusClient initialized: {len(self.metrics_to_track)} metrics from {self.base_url}")

    def _get_ray_head_node(self) -> str:
        """Get the IP address of the Ray head node where Prometheus runs.

        Returns:
            str: IP address of head node

        Raises:
            RuntimeError: If head node cannot be determined
        """
        try:
            nodes = ray.nodes()
            for node in nodes:
                if node.get("Alive") and "node:__internal_head__" in node.get("Resources", {}):
                    return node["NodeManagerAddress"]

            for node in nodes:
                if node.get("Alive") and node.get("Resources", {}).get("CPU", 0) > 0:
                    logger.warning(f"Using non-head node for Prometheus: {node['NodeManagerAddress']}")
                    return node["NodeManagerAddress"]

            raise RuntimeError("No alive Ray nodes found")

        except Exception as e:
            logger.error(f"Failed to discover Ray head node: {e}")
            return "localhost"

    def _query_metric(self, metric_name: str) -> float | None:
        """Query a single metric from Prometheus with retry logic.

        Args:
            metric_name: Prometheus metric name or query expression

        Returns:
            Metric value as float, or None if query failed
        """
        if metric_name in self._cache:
            age = time.time() - self._cache_timestamps[metric_name]
            if age < self.cache_duration:
                return self._cache[metric_name]

        url = f"{self.base_url}/api/v1/query"
        params = {"query": metric_name}

        for attempt in range(self.max_attempts):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                if data["status"] != "success":
                    logger.warning(f"Prometheus query failed: {data.get('error', 'unknown')}")
                    return None

                result = data.get("data", {}).get("result", [])
                if not result:
                    logger.debug(f"No data for metric: {metric_name}")
                    return None

                value = float(result[0]["value"][1])

                self._cache[metric_name] = value
                self._cache_timestamps[metric_name] = time.time()

                return value

            except requests.exceptions.Timeout:
                logger.warning(f"Prometheus query timeout for {metric_name} (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Prometheus connection error for {metric_name} (attempt {attempt + 1})")
            except (ValueError, KeyError, IndexError) as e:
                logger.error(f"Failed to parse Prometheus response for {metric_name}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error querying {metric_name}: {e}")
                return None

            if attempt < self.max_attempts - 1:
                time.sleep(self.retry_delay * (2**attempt))

        return None

    def query_all_metrics(self) -> dict[str, float]:
        """Query all configured metrics from Prometheus.

        Returns:
            Dictionary mapping metric names to values. Failed queries are omitted.
            Keys use 'prometheus/' prefix for namespacing in experiment tracking.
            Always returns a dict (empty if all queries fail).

        Example:
            {
                "prometheus/vllm_gpu_cache_usage_perc": 85.3,
                "prometheus/vllm_avg_generation_throughput_toks_per_s": 1247.5
            }
        """
        if not self.metrics_to_track:
            return {}

        metrics = {}
        for metric_name in self.metrics_to_track:
            try:
                value = self._query_metric(metric_name)
                if value is not None:
                    safe_name = metric_name.replace(":", "_")
                    metrics[f"prometheus/{safe_name}"] = value
            except Exception:
                pass

        return metrics

    def clear_cache(self):
        """Clear the metrics cache. Useful for testing or forced refresh."""
        self._cache.clear()
        self._cache_timestamps.clear()
