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

import copy
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from verl.workers.config import RolloutConfig
from verl.workers.config.rollout import RolloutQuantizationConfig
from verl.workers.rollout.vllm_rollout import vllm_async_server


@pytest.mark.parametrize("overrides", [None, {}, "", "{}", '{"rope_theta": 10000}', {"rope_theta": 10000}])
def test_hf_overrides_accepts_mapping_json_and_empty_values(overrides):
    """Accept both vLLM override representations without mutating the configured mapping."""
    original = copy.deepcopy(overrides)
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = RolloutConfig(name="vllm", engine_kwargs={"vllm": {"hf_overrides": overrides}})
    server.model_config = SimpleNamespace(hf_config=SimpleNamespace())

    result = server._apply_quantization()

    assert result == (json.loads(overrides) if isinstance(overrides, str) and overrides else overrides or {})
    assert overrides == original
    if isinstance(overrides, dict):
        assert result is not overrides


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("source", ["override", "checkpoint"])
@pytest.mark.parametrize("method", ["fp8", "compressed-tensors"])
def test_quantization_override_passes_through_with_exclusions(as_json, source, method):
    """Forward a quantization override verbatim, keeping both exclusion lists.

    The workers install the refit patches from vLLM's resolved ``quant_config``,
    so nothing here may rewrite the override or the checkpoint config.
    """
    quant_config = {
        "quant_method": method,
        "ignored_layers": ["model.layers.0.mlp.gate"],
        "modules_to_not_convert": ["lm_head"],
    }
    overrides = {"quantization_config": quant_config} if source == "override" else {}
    original = copy.deepcopy(overrides)
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = RolloutConfig(
        name="vllm", engine_kwargs={"vllm": {"hf_overrides": json.dumps(overrides) if as_json else overrides}}
    )
    checkpoint = quant_config if source == "checkpoint" else {"quant_method": "fp8"}
    server.model_config = SimpleNamespace(hf_config=SimpleNamespace(quantization_config=checkpoint))

    result = server._apply_quantization()

    assert result == original
    assert checkpoint == (quant_config if source == "checkpoint" else {"quant_method": "fp8"})


@pytest.mark.parametrize("as_json", [False, True])
def test_rollout_quantization_replaces_only_quantization_config(monkeypatch, as_json):
    """A rollout quantization wins over an override's quantization_config but keeps its other keys."""
    rollout_config = {"quant_method": "compressed-tensors", "config_groups": {}}
    monkeypatch.setattr(
        RolloutQuantizationConfig, "build_quantization_config", lambda self, model_path: dict(rollout_config)
    )
    overrides = {"rope_theta": 10000, "quantization_config": {"quant_method": "fp8"}}
    original = copy.deepcopy(overrides)
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = RolloutConfig(
        name="vllm",
        quantization=RolloutQuantizationConfig(linear="FP8_BLOCK"),
        engine_kwargs={"vllm": {"hf_overrides": json.dumps(overrides) if as_json else overrides}},
    )
    server.model_config = SimpleNamespace(local_path="/model", hf_config=SimpleNamespace())

    result = server._apply_quantization()

    assert result == {"rope_theta": 10000, "quantization_config": rollout_config}
    assert overrides == original
