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

"""Unit tests for normalizing an engine's adapter config for SGLang's adapter loader.

`BaseEngine.get_per_tensor_param` declares its second return value `Optional[dict]`. The
fixtures here are built by the real producers rather than hand-written, so the tests
break if either producer changes shape.
"""

from __future__ import annotations

import json

import pytest
from peft import LoraConfig, TaskType

from verl.utils.megatron_peft_utils import build_peft_config_for_vllm
from verl.workers.rollout.sglang_rollout.utils import normalize_peft_config_for_sglang

SEVEN_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _fsdp_shape() -> dict:
    """What verl/workers/engine/fsdp/transformer_impl.py returns."""
    return LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=SEVEN_PROJECTIONS,
        task_type=TaskType.CAUSAL_LM,
    ).to_dict()


def _megatron_shape() -> dict:
    """What verl/workers/engine/megatron/transformer_impl.py returns."""
    return build_peft_config_for_vllm({"rank": 32, "alpha": 32})


class TestNormalizePeftConfigForSGLang:
    def test_result_is_json_serializable(self):
        # The config crosses an HTTP boundary, so every value must survive json.dumps.
        json.dumps(normalize_peft_config_for_sglang(_fsdp_shape()))

    def test_enums_are_unwrapped_to_strings(self):
        result = normalize_peft_config_for_sglang(_fsdp_shape())
        assert result["task_type"] == "CAUSAL_LM"
        assert result["peft_type"] == "LORA"

    def test_target_modules_is_materialized_as_list(self):
        result = normalize_peft_config_for_sglang({"target_modules": {"q_proj", "v_proj"}, "peft_type": "LORA"})
        assert isinstance(result["target_modules"], list)
        assert sorted(result["target_modules"]) == ["q_proj", "v_proj"]

    def test_input_is_not_mutated(self):
        original = _fsdp_shape()
        before = dict(original)
        normalize_peft_config_for_sglang(original)
        assert original == before

    def test_missing_peft_type_is_rejected_not_guessed(self):
        # The megatron producer omits peft_type entirely. That pairing is out of scope
        # here, so it must fail with something a reader can act on rather than be
        # silently filled in with a plausible value.
        assert "peft_type" not in _megatron_shape()
        with pytest.raises(ValueError, match="peft_type"):
            normalize_peft_config_for_sglang(_megatron_shape())

    def test_rejection_names_the_producer_and_the_keys(self):
        with pytest.raises(ValueError) as excinfo:
            normalize_peft_config_for_sglang(_megatron_shape())
        message = str(excinfo.value)
        assert "build_peft_config_for_vllm" in message
        assert "target_modules" in message  # the key listing, to orient the reader
