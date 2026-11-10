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
"""dispatch_rollout_weight_quantizer takes the engine's QAT config as an object or as a plain mapping."""

from types import SimpleNamespace

import pytest

from verl.utils.qat.compressed_tensors import dispatch_rollout_weight_quantizer

FP8 = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


def _dispatch(qat_config, native=None):
    hf = SimpleNamespace(model_type="qwen3", quantization_config=native, text_config=None)
    rollout = SimpleNamespace(name="vllm", quantization=None, checkpoint_engine=SimpleNamespace(backend="naive"))
    model = SimpleNamespace(hf_config=hf, local_path="/nonexistent")
    return dispatch_rollout_weight_quantizer(model_config=model, rollout_config=rollout, qat_config=qat_config)


@pytest.mark.parametrize(
    "qat_config", [None, {}, {"enable": False, "mode": "fp8"}, SimpleNamespace(enable=False, mode="fp8")]
)
def test_qat_off_without_native_quantization_needs_no_quantizer(qat_config):
    assert _dispatch(qat_config) is None


@pytest.mark.parametrize("qat_config", [{"enable": True, "mode": "fp8"}, SimpleNamespace(enable=True, mode="fp8")])
def test_qat_on_without_a_quantized_rollout_is_refused(qat_config):
    with pytest.raises(ValueError, match="QAT needs a quantized rollout"):
        _dispatch(qat_config)


@pytest.mark.parametrize("qat_config", [{}, {"enable": True, "mode": "fp8"}, SimpleNamespace(enable=True, mode="fp8")])
def test_native_fp8_with_mapping_or_object_qat(qat_config):
    assert type(_dispatch(qat_config, native=FP8)).__name__ == "Fp8ConfigWeightQuantizer"
