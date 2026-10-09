# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Prometheus configuration updates through the active execution backend."""

import logging
import os
import subprocess

import yaml

from verl.runtime import ClassWithInitArgs, Worker, current_runtime
from verl.utils.net_utils import get_local_ip_address

logger = logging.getLogger(__name__)


class PrometheusConfigWorker(Worker):
    """Write and reload Prometheus once on each selected host."""

    def update(
        self,
        config_path: str,
        port: int,
        server_addresses: list[str],
        rollout_name: str | None,
        backend: str,
    ) -> str:
        prometheus_config = _prometheus_config(server_addresses, backend=backend)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as config_file:
            yaml.dump(prometheus_config, config_file, default_flow_style=False, indent=2)

        ip_address = get_local_ip_address().strip("[]")
        reload_url = f"http://{ip_address}:{port}/-/reload"
        try:
            subprocess.run(
                ["curl", "-X", "POST", reload_url],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:  # noqa: BLE001 - reload is best effort, matching the previous behavior
            pass
        logger.info("Reloaded Prometheus on %s", reload_url)
        return ip_address


def update_prometheus_config_with_runtime(
    config_path: str,
    port: int,
    server_addresses: list[str],
    rollout_name: str | None,
) -> None:
    """Rewrite and reload the Prometheus config on every cluster host via the current Runtime.

    Args:
        config_path: Prometheus config file written on each host.
        port: Prometheus HTTP port used for the ``/-/reload`` request.
        server_addresses: Rollout server ``host:port`` scrape targets.
        rollout_name: Rollout engine name used in log messages.
    """
    if not server_addresses:
        logger.warning("No server addresses available to update Prometheus config")
        return

    runtime = current_runtime()
    host_pool = runtime.create_resource_pool(
        nnodes=None,
        processes_per_node=1,
        device_type="cpu",
        on="cluster",
    )
    group = runtime.create_worker_group(ClassWithInitArgs(PrometheusConfigWorker), on=host_pool)
    try:
        hosts = group.remote().execute_all_sync(
            "update",
            config_path,
            port,
            server_addresses,
            rollout_name,
            runtime.backend,
        )
    finally:
        group.close()
    server_type = rollout_name.upper() if rollout_name else "rollout"
    logger.info(
        "Updated Prometheus configuration at %s on hosts %s with %d %s servers",
        config_path,
        hosts,
        len(server_addresses),
        server_type,
    )


def _prometheus_config(server_addresses: list[str], *, backend: str) -> dict:
    scrape_configs = []
    if backend == "ray":
        scrape_configs.append(
            {
                "job_name": "ray",
                "file_sd_configs": [{"files": ["/tmp/ray/prom_metrics_service_discovery.json"]}],
            }
        )
    scrape_configs.append({"job_name": "rollout", "static_configs": [{"targets": server_addresses}]})
    return {
        "global": {"scrape_interval": "10s", "evaluation_interval": "10s"},
        "scrape_configs": scrape_configs,
    }


__all__ = [
    "PrometheusConfigWorker",
    "update_prometheus_config_with_runtime",
]
