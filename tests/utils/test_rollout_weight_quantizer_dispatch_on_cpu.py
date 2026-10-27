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
"""dispatch_rollout_weight_quantizer with DeepSeek-V4, whose trainers export the checkpoint's own FP8/FP4 layout."""

from types import SimpleNamespace

import pytest

from verl.utils.qat import compressed_tensors
from verl.utils.qat.compressed_tensors import dispatch_rollout_weight_quantizer

FP8_BLOCK = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


def _configs(quantization_config, linear=None):
    hf_config = SimpleNamespace(model_type="deepseek_v4", quantization_config=quantization_config)
    quantization = SimpleNamespace(linear=linear, moe=None, ignore=[], config_file=None)
    model_config = SimpleNamespace(hf_config=hf_config, local_path=None)
    return model_config, SimpleNamespace(name="vllm", quantization=quantization)


def test_native_quantization_passes_through():
    model_config, rollout_config = _configs(FP8_BLOCK)
    quantizer = dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)
    assert quantizer.passthrough


def test_override_on_a_quantized_checkpoint_is_rejected():
    model_config, rollout_config = _configs(FP8_BLOCK, linear="FP8_BLOCK")
    with pytest.raises(NotImplementedError, match="already quantized DeepSeek-V4"):
        dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)


def test_override_on_an_unquantized_checkpoint_is_dispatched(monkeypatch):
    monkeypatch.setattr(
        compressed_tensors, "build_compressed_tensors_config", lambda **kwargs: {"quant_method": "compressed-tensors"}
    )
    monkeypatch.setattr(compressed_tensors, "CompressedTensorsWeightQuantizer", lambda config: ("quantizer", config))
    model_config, rollout_config = _configs(None, linear="FP8_BLOCK")
    quantizer = dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)
    assert quantizer == ("quantizer", {"quant_method": "compressed-tensors"})
