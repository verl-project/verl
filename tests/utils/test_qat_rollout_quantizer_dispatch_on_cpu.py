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

"""CPU tests for how vLLM ``hf_overrides`` steer the trainer-side rollout quantizer.

The rollout engine runs whatever ``quantization_config`` its ``hf_overrides``
set, so the trainer must export weights for that config rather than the
checkpoint's native one.
"""

import json
from types import SimpleNamespace

import pytest

from verl.utils.qat import compressed_tensors
from verl.utils.qat.compressed_tensors import Fp8ConfigWeightQuantizer, dispatch_rollout_weight_quantizer


def _model_config():
    # DeepSeek-V4 keeps the FP8 quantizer in passthrough mode, so no compressed-tensors install is needed.
    return SimpleNamespace(
        local_path="/model",
        hf_config=SimpleNamespace(
            model_type="deepseek_v4",
            text_config=None,
            quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
        ),
    )


def _rollout_config(hf_overrides, name="vllm"):
    return SimpleNamespace(
        name=name,
        quantization=SimpleNamespace(linear=None, moe=None, ignore=[], config_file=None),
        engine_kwargs={"vllm": {"hf_overrides": hf_overrides}},
    )


@pytest.mark.parametrize("as_json", [False, True])
def test_null_quantization_override_selects_bf16_export(as_json):
    """``hf_overrides.quantization_config=null`` runs a BF16 rollout, so nothing is quantized."""
    overrides = {"quantization_config": None}
    rollout_config = _rollout_config(json.dumps(overrides) if as_json else overrides)

    assert dispatch_rollout_weight_quantizer(model_config=_model_config(), rollout_config=rollout_config) is None


@pytest.mark.parametrize("hf_overrides", [None, {}, "", "{}", {"rope_theta": 10000}])
def test_native_quantization_without_override(hf_overrides):
    quantizer = dispatch_rollout_weight_quantizer(
        model_config=_model_config(), rollout_config=_rollout_config(hf_overrides)
    )

    assert isinstance(quantizer, Fp8ConfigWeightQuantizer)
    assert quantizer.passthrough is True


def test_vllm_override_ignored_for_other_rollout_engines():
    rollout_config = _rollout_config({"quantization_config": None}, name="sglang")

    quantizer = dispatch_rollout_weight_quantizer(model_config=_model_config(), rollout_config=rollout_config)

    assert isinstance(quantizer, Fp8ConfigWeightQuantizer)


def test_null_quantization_override_rejects_qat():
    with pytest.raises(ValueError, match="QAT needs a quantized rollout"):
        dispatch_rollout_weight_quantizer(
            model_config=_model_config(),
            rollout_config=_rollout_config({"quantization_config": None}),
            qat_config=SimpleNamespace(enable=True, mode="fp8"),
        )


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("override", [None, {"quant_method": "fp8", "source": "override"}, {}])
def test_hf_override_falls_back_to_text_config_only_for_null(monkeypatch, as_json, override):
    """Non-null overrides win; null clears only the top-level quantization config."""
    model_config = _model_config()
    text_quantization = {"quant_method": "fp8", "source": "text"}
    model_config.hf_config.text_config = SimpleNamespace(quantization_config=text_quantization)
    overrides = {"quantization_config": override}
    selected = []
    monkeypatch.setattr(
        compressed_tensors, "Fp8ConfigWeightQuantizer", lambda config, **kwargs: selected.append(config) or config
    )
    rollout_config = _rollout_config(json.dumps(overrides) if as_json else overrides)

    if override == {}:
        with pytest.raises(NotImplementedError, match="quant_method=''"):
            dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)
        assert selected == []
    else:
        expected = text_quantization if override is None else override
        result = dispatch_rollout_weight_quantizer(
            model_config=model_config,
            rollout_config=rollout_config,
            qat_config=SimpleNamespace(enable=True, mode="fp8"),
        )
        assert result == expected
        assert selected == [expected]


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("quantization", [None, SimpleNamespace(linear=None, moe=None, config_file=None)])
def test_deepseek_rejects_compressed_tensors_hf_override(as_json, model_type, nested, quantization):
    """HF overrides must not bypass DeepSeek's native mixed-layout compatibility guard."""
    model_config = _model_config()
    if nested:
        model_config.hf_config.model_type = "multimodal"
        model_config.hf_config.text_config = SimpleNamespace(model_type=model_type)
    else:
        model_config.hf_config.model_type = model_type
    overrides = {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {}}}
    rollout_config = _rollout_config(json.dumps(overrides) if as_json else overrides)
    rollout_config.quantization = quantization

    with pytest.raises(NotImplementedError, match="not supported for DeepSeek-V4"):
        dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)


def test_other_models_keep_compressed_tensors_hf_override(monkeypatch):
    model_config = _model_config()
    model_config.hf_config.model_type = "llama"
    config = {"quant_method": "compressed-tensors", "config_groups": {}}
    selected = []
    monkeypatch.setattr(
        compressed_tensors, "CompressedTensorsWeightQuantizer", lambda config: selected.append(config) or config
    )

    result = dispatch_rollout_weight_quantizer(
        model_config=model_config, rollout_config=_rollout_config({"quantization_config": config})
    )

    assert result == config
    assert selected == [config]
