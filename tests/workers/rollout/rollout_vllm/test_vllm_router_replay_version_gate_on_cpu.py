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


@pytest.mark.parametrize(
    ("layer_types", "expected"),
    [
        (None, False),
        (["full_attention", "full_attention"], False),
        (["linear_attention", "full_attention"], True),
    ],
)
def test_hybrid_routing_replay_version_gate(layer_types: list[str] | None, expected: bool) -> None:
    hf_config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(layer_types=layer_types))
    assert _hybrid_routing_replay_requires_vllm_022(hf_config, version.parse("0.18.0")) is expected


@pytest.mark.parametrize("vllm_version", ["0.22.0", "0.28.1rc1.dev93+gcacc429f6"])
def test_new_vllm_does_not_require_removed_hybrid_config_helper(monkeypatch, vllm_version: str) -> None:
    from vllm.transformers_utils import config

    monkeypatch.delattr(config, "is_interleaved", raising=False)
    assert _hybrid_routing_replay_requires_vllm_022(object(), version.parse(vllm_version)) is False
