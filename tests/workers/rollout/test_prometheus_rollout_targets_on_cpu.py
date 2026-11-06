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

import yaml

from verl.workers.rollout.utils import _get_rollout_targets


def _write_prometheus_config(path, rollout_targets):
    config = {
        "global": {"scrape_interval": "10s"},
        "scrape_configs": [
            {"job_name": "ray", "file_sd_configs": [{"files": ["/tmp/ray/prom_metrics_service_discovery.json"]}]},
            {"job_name": "rollout", "static_configs": [{"targets": rollout_targets}]},
        ],
    }
    path.write_text(yaml.safe_dump(config))


def test_merges_new_targets_with_existing_rollout_targets(tmp_path):
    config_file = tmp_path / "prometheus.yml"
    _write_prometheus_config(config_file, ["127.0.0.1:8000"])

    targets = _get_rollout_targets(str(config_file), ["127.0.0.1:8001"])

    assert targets == ["127.0.0.1:8000", "127.0.0.1:8001"]


def test_missing_config_file_returns_only_new_targets(tmp_path):
    config_file = tmp_path / "missing.yml"

    targets = _get_rollout_targets(str(config_file), ["127.0.0.1:8000"])

    assert targets == ["127.0.0.1:8000"]


def test_merge_deduplicates_targets_and_preserves_order(tmp_path):
    config_file = tmp_path / "prometheus.yml"
    _write_prometheus_config(config_file, ["127.0.0.1:8000", "127.0.0.1:8001"])

    targets = _get_rollout_targets(str(config_file), ["127.0.0.1:8001", "127.0.0.1:8002"])

    assert targets == ["127.0.0.1:8000", "127.0.0.1:8001", "127.0.0.1:8002"]


def test_malformed_config_file_returns_only_new_targets(tmp_path):
    config_file = tmp_path / "prometheus.yml"
    config_file.write_text("{invalid yaml")

    targets = _get_rollout_targets(str(config_file), ["127.0.0.1:8000"])

    assert targets == ["127.0.0.1:8000"]
