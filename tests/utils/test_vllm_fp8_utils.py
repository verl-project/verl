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

import pytest
import torch


def _import_quantize_mxfp4_weight():
    pytest.importorskip("vllm")

    try:
        from verl.utils.vllm.vllm_fp8_utils import quantize_mxfp4_weight
    except ImportError as exc:
        pytest.skip(f"vLLM FP8 utilities are unavailable: {exc}")

    return quantize_mxfp4_weight


def _import_vllm_fp8_utils():
    pytest.importorskip("vllm")

    try:
        from verl.utils.vllm import vllm_fp8_utils
    except ImportError as exc:
        pytest.skip(f"vLLM FP8 utilities are unavailable: {exc}")

    return vllm_fp8_utils


def test_quantize_mxfp4_weight_packs_two_values_per_byte():
    quantize_mxfp4_weight = _import_quantize_mxfp4_weight()

    weight = torch.arange(64, dtype=torch.float32).reshape(2, 32)

    quant_weight, quant_scale = quantize_mxfp4_weight(weight)

    assert quant_weight.dtype == torch.uint8
    assert quant_scale.dtype == torch.uint8
    assert quant_weight.shape == (2, 16)
    assert quant_scale.shape == (2, 1)


def test_quantize_mxfp4_weight_requires_32_value_blocks():
    quantize_mxfp4_weight = _import_quantize_mxfp4_weight()

    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_mxfp4_weight(torch.ones(31))


def test_prequantized_mxfp4_detection_includes_signed_packed_weights():
    vllm_fp8_utils = _import_vllm_fp8_utils()

    assert vllm_fp8_utils._is_prequantized_mxfp4_tensor(torch.empty(1, dtype=torch.int8))
    assert vllm_fp8_utils._is_prequantized_mxfp4_tensor(torch.empty(1, dtype=torch.uint8))
    assert not vllm_fp8_utils._is_prequantized_mxfp4_tensor(torch.empty(1, dtype=torch.bfloat16))


def test_prequantized_fp8_detection_keeps_loaded_fp8_weights_intact():
    vllm_fp8_utils = _import_vllm_fp8_utils()

    assert vllm_fp8_utils._is_prequantized_fp8_tensor(torch.empty(1, dtype=torch.float8_e4m3fn))
    assert not vllm_fp8_utils._is_prequantized_fp8_tensor(torch.empty(1, dtype=torch.bfloat16))


def test_deepseek_v4_scale_name_uses_sibling_scale_suffix():
    vllm_fp8_utils = _import_vllm_fp8_utils()

    class Config:
        model_type = "deepseek_v4"

    class Model:
        config = Config()

    assert (
        vllm_fp8_utils._scale_name_for_weight(
            "layers.0.ffn.experts.0.w1.weight",
            Model(),
            use_scale_not_scale_inv=True,
        )
        == "layers.0.ffn.experts.0.w1.scale"
    )


def test_model_type_handles_missing_model_and_config():
    vllm_fp8_utils = _import_vllm_fp8_utils()

    class ModelWithoutConfig:
        pass

    class ModelWithEmptyConfig:
        config = None

    assert vllm_fp8_utils._model_type(None) is None
    assert vllm_fp8_utils._model_type(ModelWithoutConfig()) is None
    assert vllm_fp8_utils._model_type(ModelWithEmptyConfig()) is None


def test_forced_scale_name_keeps_mxfp4_suffix_for_other_models():
    vllm_fp8_utils = _import_vllm_fp8_utils()

    class Model:
        pass

    assert (
        vllm_fp8_utils._scale_name_for_weight(
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            Model(),
            force_scale=True,
        )
        == "model.layers.0.mlp.experts.0.gate_proj.weight_scale"
    )


def test_default_scale_name_keeps_vllm_fp8_suffix_convention():
    vllm_fp8_utils = _import_vllm_fp8_utils()

    class Model:
        pass

    assert (
        vllm_fp8_utils._scale_name_for_weight(
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            Model(),
            use_scale_not_scale_inv=True,
        )
        == "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"
    )
