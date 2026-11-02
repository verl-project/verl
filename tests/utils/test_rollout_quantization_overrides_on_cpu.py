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

"""The trainer must export the quantization the rollout actually serves once vLLM applies hf_overrides."""

import json
from types import SimpleNamespace

import pytest

from verl.utils.qat.compressed_tensors import (
    Fp8ConfigWeightQuantizer,
    _rollout_quantization_config,
    dispatch_rollout_weight_quantizer,
)

_FP8 = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
_CT = {"quant_method": "compressed-tensors", "config_groups": {}}


def _multimodal_fp8():
    """A multimodal checkpoint whose FP8 config lives on the text config only."""
    return SimpleNamespace(
        model_type="vl", quantization_config=None, text_config=SimpleNamespace(quantization_config=_FP8)
    )


def _rollout(hf_overrides=None, name="vllm"):
    engine_kwargs = {} if hf_overrides is None else {"vllm": {"hf_overrides": hf_overrides}}
    return SimpleNamespace(
        name=name,
        engine_kwargs=engine_kwargs,
        quantization=None,
        checkpoint_engine=SimpleNamespace(backend="naive"),
    )


@pytest.mark.parametrize(
    ("hf_overrides", "expected"),
    [
        (None, (False, _FP8)),
        # A nested null clears the text-level config: vLLM then serves an unquantized rollout.
        ({"text_config": {"quantization_config": None}}, (True, None)),
        # A top-level null leaves vLLM's fallback to the text-level config in place.
        ({"quantization_config": None}, (True, _FP8)),
        ({"quantization_config": None, "text_config": {"quantization_config": None}}, (True, None)),
        # A nested mapping merges into the checkpoint's config key by key.
        (
            {"text_config": {"quantization_config": {"activation_scheme": "static"}}},
            (True, {**_FP8, "activation_scheme": "static"}),
        ),
        # A top-level config wins over the text-level one.
        ({"quantization_config": _CT, "text_config": {"quantization_config": None}}, (True, _CT)),
        # Other text_config overrides leave the quantization alone.
        ({"text_config": {"rope_theta": 1.0}}, (False, _FP8)),
        (json.dumps({"text_config": {"quantization_config": None}}), (True, None)),
    ],
)
def test_rollout_quantization_config_follows_vllm_hf_overrides(hf_overrides, expected):
    assert _rollout_quantization_config(_multimodal_fp8(), _rollout(hf_overrides)) == expected


def test_hf_overrides_apply_only_to_vllm():
    rollout = _rollout({"text_config": {"quantization_config": None}}, name="sglang")
    assert _rollout_quantization_config(_multimodal_fp8(), rollout) == (False, _FP8)


def test_text_config_override_needs_a_text_config():
    hf_config = SimpleNamespace(quantization_config=None)
    rollout = _rollout({"text_config": {"quantization_config": _FP8}})
    assert _rollout_quantization_config(hf_config, rollout) == (False, None)


def test_dispatch_exports_unquantized_weights_when_hf_overrides_null_the_text_config():
    pytest.importorskip("compressed_tensors")
    model_config = SimpleNamespace(hf_config=_multimodal_fp8(), local_path=None)

    native = dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=_rollout())
    assert isinstance(native, Fp8ConfigWeightQuantizer)

    overridden = dispatch_rollout_weight_quantizer(
        model_config=model_config,
        rollout_config=_rollout({"text_config": {"quantization_config": None}}),
    )
    assert overridden is None


# DeepSeek-V4.1-Flash's checkpoint quantization_config.
_DSV4_FP8 = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def _deepseek_v4(model_type):
    """DeepSeek-V4 keeps its quantization_config at the top level, with no text config."""
    hf_config = SimpleNamespace(model_type=model_type, quantization_config=_DSV4_FP8)
    return SimpleNamespace(hf_config=hf_config, local_path=None)


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize(
    ("hf_overrides", "change"),
    [
        ({"quantization_config": None}, "cleared it"),
        (json.dumps({"quantization_config": None}), "cleared it"),
        ({"quantization_config": _CT}, "quant_method"),
        # Without weight_block_size vLLM allocates a per-tensor weight_scale, not the blockwise weight_scale_inv.
        (
            {"quantization_config": {k: v for k, v in _DSV4_FP8.items() if k != "weight_block_size"}},
            "changed weight_block_size",
        ),
        ({"quantization_config": {**_DSV4_FP8, "scale_fmt": None}}, "changed scale_fmt"),
    ],
)
def test_dispatch_rejects_hf_overrides_that_change_deepseek_v4_quantization(model_type, hf_overrides, change):
    # Megatron-Bridge still exports the checkpoint's FP8/FP4 layout, which only its own config loads.
    with pytest.raises(NotImplementedError, match="may not change DeepSeek-V4's quantization_config") as exc_info:
        dispatch_rollout_weight_quantizer(model_config=_deepseek_v4(model_type), rollout_config=_rollout(hf_overrides))
    assert change in str(exc_info.value)


@pytest.mark.parametrize(
    "hf_overrides",
    [
        None,
        {"text_config": {"quantization_config": None}},
        {"quantization_config": dict(_DSV4_FP8)},
        json.dumps({"quantization_config": _DSV4_FP8}),
    ],
)
def test_dispatch_passes_deepseek_v4_native_quantization_through(hf_overrides):
    quantizer = dispatch_rollout_weight_quantizer(
        model_config=_deepseek_v4("deepseek_v4"), rollout_config=_rollout(hf_overrides)
    )
    assert isinstance(quantizer, Fp8ConfigWeightQuantizer)
    assert quantizer.passthrough
