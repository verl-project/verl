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
"""dispatch_rollout_weight_quantizer: combinations the trainer-side rollout weight quantizer cannot serve.

DeepSeek-V4 trainers export the checkpoint's own names and FP8/FP4 layout; delta_sharded steady-state syncs
bypass the quantizer.
"""

from types import SimpleNamespace

import pytest

from verl.utils.qat import compressed_tensors
from verl.utils.qat.compressed_tensors import dispatch_rollout_weight_quantizer

FP8_BLOCK = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


def _configs(quantization_config, linear=None, moe=None, model_type="deepseek_v4", checkpoint_backend="naive"):
    hf_config = SimpleNamespace(model_type=model_type, quantization_config=quantization_config)
    quantization = SimpleNamespace(linear=linear, moe=moe, ignore=[], config_file=None)
    model_config = SimpleNamespace(hf_config=hf_config, local_path=None)
    checkpoint_engine = SimpleNamespace(backend=checkpoint_backend)
    return model_config, SimpleNamespace(name="vllm", quantization=quantization, checkpoint_engine=checkpoint_engine)


def test_native_deepseek_v4_quantization_passes_through():
    model_config, rollout_config = _configs(FP8_BLOCK)
    quantizer = dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)
    assert quantizer.passthrough


@pytest.mark.parametrize("native", [FP8_BLOCK, None], ids=["quantized", "unquantized"])
@pytest.mark.parametrize("override", [{"linear": "FP8_BLOCK"}, {"moe": "MXFP8"}], ids=["linear", "moe"])
def test_overrides_are_rejected_for_deepseek_v4(native, override):
    model_config, rollout_config = _configs(native, **override)
    with pytest.raises(NotImplementedError, match="not supported for DeepSeek-V4"):
        dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)


def test_override_on_another_model_is_dispatched(monkeypatch):
    monkeypatch.setattr(
        compressed_tensors, "build_compressed_tensors_config", lambda **kwargs: {"quant_method": "compressed-tensors"}
    )
    monkeypatch.setattr(compressed_tensors, "CompressedTensorsWeightQuantizer", lambda config: ("quantizer", config))
    model_config, rollout_config = _configs(None, linear="FP8_BLOCK", model_type="qwen3_moe")
    quantizer = dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)
    assert quantizer == ("quantizer", {"quant_method": "compressed-tensors"})


def test_delta_sharded_sync_is_rejected_for_a_quantized_rollout():
    model_config, rollout_config = _configs(FP8_BLOCK, checkpoint_backend="delta_sharded")
    with pytest.raises(NotImplementedError, match="delta_sharded"):
        dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)
    model_config, rollout_config = _configs(None, checkpoint_backend="delta_sharded")
    assert dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config) is None
