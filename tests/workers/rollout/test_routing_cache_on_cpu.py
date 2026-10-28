# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# This dependency-free policy does not need the package's vLLM runtime bootstrap.
_path = Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout/routing_cache.py"
_spec = importlib.util.spec_from_file_location("routing_cache_cpu", _path)
_module = importlib.util.module_from_spec(_spec)
with patch.dict(sys.modules, {_spec.name: _module}):
    _spec.loader.exec_module(_module)
RoutingCachePolicy = _module.RoutingCachePolicy


@pytest.mark.parametrize("aux_output", [False, True])
@pytest.mark.parametrize("routing_replay", [False, True])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_cache_policy_preserves_explicit_reuse_choice(aux_output, routing_replay, prefix_caching):
    engine_args = SimpleNamespace(__dataclass_fields__={"aux_output_config": None} if aux_output else {})
    policy = RoutingCachePolicy.from_config(
        prefix_caching=prefix_caching, routing_replay=routing_replay, engine_args_type=engine_args
    )
    first = {"prompt_token_ids": [1, 2, 3]}
    retry = dict(first)
    policy.prepare_prompt(first)
    policy.prepare_prompt(retry)
    if aux_output and routing_replay and not prefix_caching:
        assert policy.prefix_caching
        assert first["cache_salt"] != retry["cache_salt"]
    else:
        assert policy.prefix_caching == prefix_caching
        assert "cache_salt" not in first and "cache_salt" not in retry
    assert first["prompt_token_ids"] == retry["prompt_token_ids"] == [1, 2, 3]
