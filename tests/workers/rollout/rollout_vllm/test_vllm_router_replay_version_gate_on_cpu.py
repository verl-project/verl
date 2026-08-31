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

from types import SimpleNamespace

import pytest
from packaging import version

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout.vllm_async_server import _hybrid_routing_replay_requires_vllm_022


class _ModelConfig:
    def __init__(self, layer_types: list[str] | None):
        self._text_config = SimpleNamespace(layer_types=layer_types)

    def get_text_config(self) -> SimpleNamespace:
        return self._text_config


@pytest.mark.parametrize(
    ("layer_types", "vllm_version", "expected"),
    [
        (None, "0.18.0", False),
        (["full_attention", "full_attention"], "0.18.0", False),
        (["linear_attention", "full_attention"], "0.18.0", True),
        (["linear_attention", "full_attention"], "0.22.0", False),
    ],
)
def test_hybrid_routing_replay_version_gate(layer_types: list[str] | None, vllm_version: str, expected: bool) -> None:
    assert _hybrid_routing_replay_requires_vllm_022(_ModelConfig(layer_types), version.parse(vllm_version)) is expected
