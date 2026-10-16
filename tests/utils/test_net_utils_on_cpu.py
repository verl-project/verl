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

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[2] / "verl" / "utils" / "net_utils.py"
_SPEC = importlib.util.spec_from_file_location("verl_net_utils", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
net_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(net_utils)


def test_ranked_port_base_is_stable_and_in_range():
    port = net_utils.get_ranked_port_base("job-123:rollout", 7)

    assert port == net_utils.get_ranked_port_base("job-123:rollout", 7)
    assert 20000 <= port <= 60000


def test_ranked_port_base_separates_colocated_vllm_engines():
    namespaces = ("job-123:rollout", "job-123:reward", "job-123:teacher")
    ports = {net_utils.get_ranked_port_base(namespace, rank) for namespace in namespaces for rank in range(4)}

    assert len(ports) == len(namespaces) * 4


def test_ranked_port_base_applies_rank_stride_with_wraparound():
    min_port = 20000
    max_port = 20010
    rank_stride = 7
    base_port = net_utils.get_ranked_port_base(
        "job-123:rollout",
        0,
        min_port=min_port,
        max_port=max_port,
        rank_stride=rank_stride,
    )

    assert net_utils.get_ranked_port_base(
        "job-123:rollout",
        12,
        min_port=min_port,
        max_port=max_port,
        rank_stride=rank_stride,
    ) == min_port + (base_port - min_port + 12 * rank_stride) % (max_port - min_port + 1)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"namespace": "", "rank": 0}, "namespace must be non-empty"),
        ({"namespace": "rollout", "rank": -1}, "rank must be non-negative"),
        ({"namespace": "rollout", "rank": 0, "min_port": 1023}, "invalid TCP port range"),
        ({"namespace": "rollout", "rank": 0, "min_port": 30000, "max_port": 29999}, "invalid TCP port range"),
        ({"namespace": "rollout", "rank": 0, "max_port": 65536}, "invalid TCP port range"),
        ({"namespace": "rollout", "rank": 0, "rank_stride": 0}, "rank_stride must be positive"),
    ],
)
def test_ranked_port_base_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        net_utils.get_ranked_port_base(**kwargs)
