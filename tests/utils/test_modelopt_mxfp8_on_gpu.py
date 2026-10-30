# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("compressed_tensors")
pytest.importorskip("modelopt")

# Hide an unusable optional FlashAttention-4 package before ModelOpt imports Megatron.
# isort: off
import verl.models.mcore  # noqa: F401
import modelopt.torch.quantization as mtq
from verl.utils.qat.modelopt import (
    _Mxfp4FakeQuant,
    _Nvfp4FakeQuant,
    _nvfp4_global_scale,
    _nvfp4_input_backend,
    _nvfp4_weight_backend,
    build_quantize_config,
)
from verl.utils.qat.compressed_tensors import (
    CompressedTensorsWeightQuantizer,
    Fp8ConfigWeightQuantizer,
    _compress_weight,
    _scheme_for_preset,
    build_compressed_tensors_config,
    dispatch_rollout_weight_quantizer,
    qat_modes_for_config,
)
# isort: on


def _entry(config: dict, name: str):
    return next(entry for entry in config["quant_cfg"] if entry["quantizer_name"] == name)


def test_mxfp8_qat_uses_modelopt_native_preset():
    config = build_quantize_config("mxfp8")
    expected = mtq.normalize_quant_cfg_list(mtq.MXFP8_DEFAULT_CFG["quant_cfg"])

    assert config["quant_cfg"] == expected
    assert config["algorithm"] is None
    weight_cfg = _entry(config, "*weight_quantizer")["cfg"]
    assert weight_cfg["num_bits"] == (4, 3)
    assert weight_cfg["block_sizes"] == {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}
    assert weight_cfg.get("backend") is None


def test_mxfp8_qat_appends_user_ignores_without_mutating_modelopt_preset():
    original = mtq.normalize_quant_cfg_list(mtq.MXFP8_DEFAULT_CFG["quant_cfg"])

    config = build_quantize_config("mxfp8", ["*linear_attn.in_proj_a"])

    model = torch.nn.Module()
    model.linear_attn = torch.nn.Module()
    model.linear_attn.in_proj_a = torch.nn.Linear(128, 32, bias=False)
    model.linear_attn.in_proj_ab = torch.nn.Linear(128, 128, bias=False)
    mtq.quantize(model, config)

    assert not model.linear_attn.in_proj_a.input_quantizer.is_enabled
    assert not model.linear_attn.in_proj_a.weight_quantizer.is_enabled
    assert model.linear_attn.in_proj_ab.input_quantizer.is_enabled
    assert model.linear_attn.in_proj_ab.weight_quantizer.is_enabled
    assert mtq.normalize_quant_cfg_list(mtq.MXFP8_DEFAULT_CFG["quant_cfg"]) == original


@pytest.mark.parametrize("mode", ["fp8", "mxfp8"])
@pytest.mark.parametrize(
    "pattern, input_enabled, weight_enabled",
    [
        ("projection", False, False),
        ("projectio?", False, False),
        ("projection.*", False, False),
        ("projection.weight_quantizer", True, False),
        ("*input_quantizer", False, True),
    ],
)
def test_qat_ignore_module_and_quantizer_patterns(mode, pattern, input_enabled, weight_enabled):
    model = torch.nn.Module()
    model.projection = torch.nn.Linear(128, 128, bias=False)
    mtq.quantize(model, build_quantize_config(mode, [pattern]))

    assert model.projection.input_quantizer.is_enabled == input_enabled
    assert model.projection.weight_quantizer.is_enabled == weight_enabled


@pytest.mark.parametrize(
    "pattern, module_name", [("lm_head", "output_layer"), ("*mlp.gate", "router"), ("*self_attn*", "self_attention")]
)
def test_qat_ignore_keeps_megatron_aliases(pattern, module_name):
    model = torch.nn.Module()
    model.add_module(module_name, torch.nn.Linear(128, 128, bias=False))
    mtq.quantize(model, build_quantize_config("fp8", [pattern]))

    layer = model.get_submodule(module_name)
    assert not layer.input_quantizer.is_enabled
    assert not layer.weight_quantizer.is_enabled


def test_fp8_ignored_module_preserves_forward_and_backward():
    torch.manual_seed(7)
    model = torch.nn.Module()
    model.projection = torch.nn.Linear(128, 32, bias=False).to(torch.bfloat16)
    reference = copy.deepcopy(model.projection)
    mtq.quantize(model, build_quantize_config("fp8", ["projection"]))
    inputs = torch.randn(16, 128, dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)

    actual = model.projection(inputs)
    expected = reference(reference_inputs)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.projection.weight.grad, reference.weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)


def test_mxfp8_rollout_requires_explicit_shape_ignore():
    pattern = r"re:.*linear_attn\.in_proj_[ab]$"

    config = build_compressed_tensors_config(
        linear="MXFP8",
        moe="MXFP8",
        ignore=[pattern],
        config_file=None,
        # Config construction must not inspect the checkpoint.
        model_path="/path/does/not/exist",
    )

    assert pattern in config["ignore"]


def test_mxfp4_rollout_is_moe_only_w4a8_mx():
    config = build_compressed_tensors_config(
        linear=None,
        moe="MXFP4",
        ignore=[],
        config_file=None,
    )

    scheme = config["config_groups"]["group_moe"]
    assert scheme["format"] == "mxfp4-pack-quantized"
    assert scheme["weights"]["num_bits"] == 4
    assert scheme["weights"]["group_size"] == 32
    assert scheme["weights"]["scale_dtype"] == "torch.uint8"
    assert scheme["input_activations"]["num_bits"] == 8
    assert scheme["input_activations"]["group_size"] == 32
    assert scheme["input_activations"]["dynamic"] is True
    assert qat_modes_for_config(config) == {"mxfp4"}

    with pytest.raises(ValueError, match="only for routed MoE experts"):
        build_compressed_tensors_config(
            linear="MXFP4",
            moe=None,
            ignore=[],
            config_file=None,
        )


def test_mxfp4_rollout_quantizes_packed_experts():
    config = build_compressed_tensors_config(
        linear=None,
        moe="MXFP4",
        ignore=[],
        config_file=None,
    )
    quantizer = CompressedTensorsWeightQuantizer(config)
    weights = torch.linspace(-1, 1, 2 * 64 * 32, dtype=torch.bfloat16).reshape(2, 64, 32)

    result = dict(quantizer([("model.layers.0.mlp.experts.gate_up_proj", weights)]))

    assert set(result) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight_packed",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.0.mlp.experts.0.up_proj.weight_packed",
        "model.layers.0.mlp.experts.0.up_proj.weight_scale",
        "model.layers.0.mlp.experts.1.gate_proj.weight_packed",
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale",
        "model.layers.0.mlp.experts.1.up_proj.weight_packed",
        "model.layers.0.mlp.experts.1.up_proj.weight_scale",
    }
    assert result["model.layers.0.mlp.experts.0.gate_proj.weight_packed"].dtype == torch.uint8
    assert result["model.layers.0.mlp.experts.0.gate_proj.weight_packed"].shape == (32, 16)
    assert result["model.layers.0.mlp.experts.0.gate_proj.weight_scale"].dtype == torch.uint8
    assert result["model.layers.0.mlp.experts.0.gate_proj.weight_scale"].shape == (32, 1)


def test_native_fp8_config_export_uses_vllm_scale_names():
    quantizer = Fp8ConfigWeightQuantizer(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["model.layers.0.mlp.gate"],
        }
    )
    weight = torch.linspace(-1, 1, 256 * 256, dtype=torch.bfloat16).reshape(256, 256)
    gate = torch.randn(16, 256, dtype=torch.bfloat16)

    result = dict(
        quantizer(
            [
                ("model.layers.0.self_attn.q_proj.weight", weight),
                ("model.layers.0.mlp.gate.weight", gate),
            ]
        )
    )

    assert set(result) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale_inv",
        "model.layers.0.mlp.gate.weight",
    }
    assert result["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    assert result["model.layers.0.self_attn.q_proj.weight_scale_inv"].shape == (2, 2)
    assert result["model.layers.0.mlp.gate.weight"] is gate


def test_native_fp8_config_export_keeps_default_ignored_layers_in_high_precision():
    quantizer = Fp8ConfigWeightQuantizer(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    )
    embedding = torch.randn(256, 256, dtype=torch.bfloat16)
    lm_head = torch.randn(256, 256, dtype=torch.bfloat16)
    router = torch.randn(16, 256, dtype=torch.bfloat16)

    result = dict(
        quantizer(
            [
                ("model.embed_tokens.weight", embedding),
                ("lm_head.weight", lm_head),
                ("model.layers.0.mlp.gate.weight", router),
            ]
        )
    )

    expected = {
        "model.embed_tokens.weight": embedding,
        "lm_head.weight": lm_head,
        "model.layers.0.mlp.gate.weight": router,
    }
    assert result.keys() == expected.keys()
    assert all(result[name] is tensor for name, tensor in expected.items())


def test_native_fp8_config_export_unpacks_fused_experts():
    quantizer = Fp8ConfigWeightQuantizer(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    )
    weights = torch.randn(2, 256, 128, dtype=torch.bfloat16)

    result = dict(quantizer([("model.layers.0.mlp.experts.down_proj", weights)]))

    assert set(result) == {
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.1.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight_scale_inv",
    }
    assert result["model.layers.0.mlp.experts.0.down_proj.weight"].dtype == torch.float8_e4m3fn


def _rollout_config(**quantization):
    return SimpleNamespace(
        quantization=SimpleNamespace(
            linear=quantization.get("linear"),
            moe=quantization.get("moe"),
            ignore=quantization.get("ignore", []),
            config_file=quantization.get("config_file"),
        )
    )


def test_weight_quantizer_dispatches_native_fp8_and_deepseek_passthrough():
    fp8_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    standard_model = SimpleNamespace(
        local_path="/model",
        hf_config=SimpleNamespace(model_type="glm_moe_dsa", text_config=None, quantization_config=fp8_config),
    )
    deepseek_model = SimpleNamespace(
        local_path="/model",
        hf_config=SimpleNamespace(
            model_type="deepseek_v4", text_config=None, quantization_config={"quant_method": "fp8"}
        ),
    )

    standard = dispatch_rollout_weight_quantizer(model_config=standard_model, rollout_config=_rollout_config())
    deepseek = dispatch_rollout_weight_quantizer(model_config=deepseek_model, rollout_config=_rollout_config())

    assert isinstance(standard, Fp8ConfigWeightQuantizer)
    assert standard.passthrough is False
    assert isinstance(deepseek, Fp8ConfigWeightQuantizer)
    assert deepseek.passthrough is True
    weight = torch.tensor([-1, 2], dtype=torch.int8)
    scale = torch.tensor([127, -1], dtype=torch.float8_e8m0fnu)
    dense = torch.ones(2, dtype=torch.bfloat16)
    result = dict(
        deepseek(
            [
                ("model.layers.0.mlp.experts.0.weight", weight),
                ("model.layers.0.mlp.experts.0.scale", scale),
                ("model.layers.0.self_attn.q_proj.weight", dense),
            ]
        )
    )
    assert result["model.layers.0.mlp.experts.0.weight"].dtype == torch.uint8
    assert result["model.layers.0.mlp.experts.0.weight"].data_ptr() == weight.data_ptr()
    assert result["model.layers.0.mlp.experts.0.scale"].dtype == torch.uint8
    assert result["model.layers.0.mlp.experts.0.scale"].data_ptr() == scale.data_ptr()
    assert result["model.layers.0.self_attn.q_proj.weight"] is dense


def test_weight_quantizer_dispatches_native_and_override_compressed_tensors():
    native_config = build_compressed_tensors_config(linear="FP8_BLOCK", moe="FP8_BLOCK", ignore=[], config_file=None)
    native_model = SimpleNamespace(
        local_path="/model",
        hf_config=SimpleNamespace(model_type="qwen3_5_moe", text_config=None, quantization_config=native_config),
    )
    bf16_model = SimpleNamespace(
        local_path="/model",
        hf_config=SimpleNamespace(model_type="qwen3_5_moe", text_config=None, quantization_config=None),
    )

    native = dispatch_rollout_weight_quantizer(model_config=native_model, rollout_config=_rollout_config())
    override = dispatch_rollout_weight_quantizer(
        model_config=bf16_model,
        rollout_config=_rollout_config(linear="FP8_BLOCK", moe="FP8_BLOCK"),
    )

    assert isinstance(native, CompressedTensorsWeightQuantizer)
    assert isinstance(override, CompressedTensorsWeightQuantizer)


def test_mixed_mxfp8_linear_mxfp4_moe_uses_composite_qat_mode():
    config = build_compressed_tensors_config(
        linear="MXFP8",
        moe="MXFP4",
        ignore=[],
        config_file=None,
    )

    assert config["format"] == "mixed-precision"
    assert qat_modes_for_config(config) == {"mxfp4_mxfp8"}


@pytest.mark.parametrize(
    ("qat_mode", "dense_mxfp8"),
    [("mxfp4", False), ("mxfp4_mxfp8", True)],
)
def test_mxfp4_qat_combines_mxfp4_weights_with_mxfp8_inputs(qat_mode, dense_mxfp8):
    config = build_quantize_config(qat_mode)
    weight_cfg = _entry(config, "*mlp.experts*weight_quantizer")["cfg"]
    input_cfg = _entry(config, "*mlp.experts*input_quantizer")["cfg"]

    assert weight_cfg["num_bits"] == (2, 1)
    assert weight_cfg["block_sizes"] == {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}
    assert weight_cfg["backend"] == "verl_mxfp4"
    assert input_cfg["num_bits"] == (4, 3)
    assert input_cfg["block_sizes"] == {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}
    assert config["algorithm"] is None

    generic_weight_entries = [entry for entry in config["quant_cfg"] if entry["quantizer_name"] == "*weight_quantizer"]
    assert bool(generic_weight_entries) is dense_mxfp8


def test_mxfp4_qat_weight_qdq_exactly_matches_export():
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.compressors.format import infer_module_format

    weight = torch.randn(64, 64, dtype=torch.bfloat16)
    fake_quantized = _Mxfp4FakeQuant.apply(weight)
    scheme = _scheme_for_preset("MXFP4", ["Linear"])
    compressed = _compress_weight(weight, scheme)
    compressor = BaseCompressor.get_value_from_registry(infer_module_format(torch.nn.Linear, scheme).value)
    exported_qdq = compressor.decompress(compressed, scheme)["weight"]

    torch.testing.assert_close(fake_quantized, exported_qdq, rtol=0, atol=0)


def test_nvfp4_rollout_is_moe_only_w4a4():
    config = build_compressed_tensors_config(
        linear=None,
        moe="NVFP4",
        ignore=[],
        config_file=None,
    )

    scheme = config["config_groups"]["group_moe"]
    assert config["format"] == "nvfp4-pack-quantized"
    assert scheme["weights"]["strategy"] == "tensor_group"
    assert scheme["weights"]["group_size"] == 16
    assert scheme["weights"]["scale_dtype"] == "torch.float8_e4m3fn"
    assert scheme["input_activations"]["strategy"] == "tensor_group"
    assert scheme["input_activations"]["dynamic"] == "local"
    assert qat_modes_for_config(config) == {"nvfp4"}

    with pytest.raises(ValueError, match="only for routed MoE experts"):
        build_compressed_tensors_config(
            linear="NVFP4",
            moe=None,
            ignore=[],
            config_file=None,
        )


def test_nvfp4_qat_uses_experts_only_export_compatible_backends():
    config = build_quantize_config("nvfp4")
    weight_cfg = _entry(config, "*mlp.experts*weight_quantizer")["cfg"]
    input_cfg = _entry(config, "*mlp.experts*input_quantizer")["cfg"]

    assert weight_cfg["backend"] == "verl_nvfp4_weight"
    assert input_cfg["backend"] == "verl_nvfp4_input"
    assert weight_cfg["block_sizes"] == {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
    assert input_cfg["block_sizes"] == {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
    assert config["algorithm"] is None
    assert not any(
        entry["quantizer_name"] == "*weight_quantizer" and entry.get("enable", True) for entry in config["quant_cfg"]
    )


def test_nvfp4_qat_weight_qdq_exactly_matches_export():
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.compressors.format import infer_module_format

    weight = torch.randn(64, 64, dtype=torch.bfloat16)
    global_scale = _nvfp4_global_scale(weight)
    fake_quantized = _Nvfp4FakeQuant.apply(weight, global_scale)
    scheme = _scheme_for_preset("NVFP4", ["Linear"])
    compressed = _compress_weight(weight, scheme, global_scale=global_scale)
    compressor = BaseCompressor.get_value_from_registry(infer_module_format(torch.nn.Linear, scheme).value)
    exported_qdq = compressor.decompress(compressed, scheme)["weight"]

    torch.testing.assert_close(fake_quantized, exported_qdq, rtol=0, atol=0)


def test_nvfp4_grouped_expert_qdq_uses_one_global_scale_per_expert():
    weight = torch.randn(3, 64, 32, dtype=torch.bfloat16)
    actual = _nvfp4_weight_backend(weight, None)
    expected = torch.stack([_Nvfp4FakeQuant.apply(expert, _nvfp4_global_scale(expert)) for expert in weight])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvfp4_input_backend_initializes_from_first_observation_then_tracks_minimum():
    quantizer = SimpleNamespace()
    first = torch.full((16, 16), 0.5, dtype=torch.bfloat16)
    second = first * 2

    _nvfp4_input_backend(first, quantizer)
    first_scale = _nvfp4_global_scale(first)
    torch.testing.assert_close(quantizer._verl_input_global_scale, first_scale)

    _nvfp4_input_backend(second, quantizer)
    torch.testing.assert_close(
        quantizer._verl_input_global_scale,
        torch.minimum(first_scale, _nvfp4_global_scale(second)),
    )


def test_nvfp4_export_uses_neutral_then_observed_input_scales():
    config = build_compressed_tensors_config(
        linear=None,
        moe="NVFP4",
        ignore=[],
        config_file=None,
    )
    weights = torch.randn(2, 64, 32, dtype=torch.bfloat16)
    quantizer = CompressedTensorsWeightQuantizer(config)
    neutral = dict(quantizer([("model.layers.0.mlp.experts.gate_up_proj", weights)]))
    assert neutral["model.layers.0.mlp.experts.0.gate_proj.input_global_scale"].item() == 1.0

    quantizer.set_input_global_scales(
        {
            "layers.0.mlp.experts.gate_up": torch.tensor([0.25]),
            "layers.0.mlp.experts.down": torch.tensor([0.5]),
        }
    )
    result = dict(quantizer([("model.layers.0.mlp.experts.gate_up_proj", weights)]))
    gate_prefix = "model.layers.0.mlp.experts.0.gate_proj."
    up_prefix = "model.layers.0.mlp.experts.0.up_proj."
    assert result[gate_prefix + "input_global_scale"].item() == 0.25
    assert result[up_prefix + "input_global_scale"].item() == 0.25
    torch.testing.assert_close(
        result[gate_prefix + "weight_global_scale"],
        result[up_prefix + "weight_global_scale"],
    )


@pytest.mark.parametrize("layer_numbers", [((1, 2),), ((5, 6),), ((1, 2), (5, 6)), ((5, 6), (9, 10))])
@pytest.mark.parametrize("wrapped", [False, True])
def test_nvfp4_activation_scales_use_global_pipeline_layer_numbers(layer_numbers, wrapped):
    from verl.utils.qat.modelopt import collect_nvfp4_input_global_scales

    chunks = []
    expected = {}
    for numbers in layer_numbers:
        chunk = torch.nn.Module()
        chunk.decoder = torch.nn.Module()
        chunk.decoder.layers = torch.nn.ModuleList()
        for number in numbers:
            layer = torch.nn.Module()
            layer.layer_number = number
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = torch.nn.Module()
            for projection, key in (("linear_fc1", "gate_up"), ("linear_fc2", "down")):
                linear = torch.nn.Module()
                linear.input_quantizer = torch.nn.Module()
                scale = torch.tensor([float(number * (2 if key == "down" else 1))])
                linear.input_quantizer._verl_input_global_scale = scale
                setattr(layer.mlp.experts, projection, linear)
                expected[f"layers.{number - 1}.mlp.experts.{key}"] = scale
            chunk.decoder.layers.append(layer)
        if wrapped:
            wrapper = torch.nn.Sequential(chunk)
            chunk = wrapper
        chunks.append(chunk)

    actual = collect_nvfp4_input_global_scales(chunks)
    assert actual.keys() == expected.keys()
    for key, scale in expected.items():
        torch.testing.assert_close(actual[key], scale, rtol=0, atol=0)

    config = build_compressed_tensors_config(linear=None, moe="NVFP4", ignore=[], config_file=None)
    quantizer = CompressedTensorsWeightQuantizer(config)
    quantizer.refresh_input_global_scales(chunks)
    for numbers in layer_numbers:
        for number in numbers:
            prefix = f"model.layers.{number - 1}.mlp.experts"
            weights = torch.ones(2, 64, 32, dtype=torch.bfloat16)
            result = dict(quantizer([(f"{prefix}.gate_up_proj", weights), (f"{prefix}.down_proj", weights)]))
            for expert in range(2):
                for projection, key in (("gate_proj", "gate_up"), ("up_proj", "gate_up"), ("down_proj", "down")):
                    torch.testing.assert_close(
                        result[f"{prefix}.{expert}.{projection}.input_global_scale"],
                        expected[f"layers.{number - 1}.mlp.experts.{key}"],
                        rtol=0,
                        atol=0,
                    )


def _nvfp4_pipeline_scale_worker(rank, init_file, virtual_chunks):
    import torch.distributed as dist

    from verl.utils.qat.modelopt import collect_nvfp4_input_global_scales

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        chunks = []
        observers = []
        for vp in range(virtual_chunks):
            chunk = torch.nn.Module()
            chunk.decoder = torch.nn.Module()
            layer = torch.nn.Module()
            layer.layer_number = vp * 2 + rank + 1
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = torch.nn.Module()
            layer.mlp.experts.linear_fc1 = torch.nn.Module()
            observer = torch.nn.Module()
            layer.mlp.experts.linear_fc1.input_quantizer = observer
            chunk.decoder.layers = torch.nn.ModuleList([layer])
            chunks.append(chunk)
            observers.append((layer.layer_number, observer))

        assert collect_nvfp4_input_global_scales(chunks, pp_group=dist.group.WORLD) == {}
        config = build_compressed_tensors_config(linear=None, moe="NVFP4", ignore=[], config_file=None)
        quantizer = CompressedTensorsWeightQuantizer(config)
        for step in (1, 2):
            for number, observer in observers:
                observer._verl_input_global_scale = torch.tensor([float(number * step)])
            quantizer.refresh_input_global_scales(chunks, pp_group=dist.group.WORLD)
            for layer in range(virtual_chunks * 2):
                prefix = f"model.layers.{layer}.mlp.experts"
                output = dict(quantizer([(f"{prefix}.gate_up_proj", torch.ones(2, 64, 32))]))
                for expert in range(2):
                    for projection in ("gate_proj", "up_proj"):
                        torch.testing.assert_close(
                            output[f"{prefix}.{expert}.{projection}.input_global_scale"],
                            torch.tensor([float((layer + 1) * step)]),
                            rtol=0,
                            atol=0,
                        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("virtual_chunks", [1, 2])
def test_nvfp4_pipeline_scales_reach_every_export_rank(tmp_path, virtual_chunks):
    torch.multiprocessing.spawn(
        _nvfp4_pipeline_scale_worker,
        args=(str(tmp_path / "init"), virtual_chunks),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("expert_indices", [(0, 1), (4, 5)])
def test_nvfp4_sequential_experts_keep_individual_scales(expert_indices):
    from verl.utils.qat.modelopt import collect_nvfp4_input_global_scales

    chunk = torch.nn.Module()
    chunk.decoder = torch.nn.Module()
    layer = torch.nn.Module()
    layer.layer_number = 5
    layer.mlp = torch.nn.Module()
    layer.mlp.local_expert_indices = expert_indices
    layer.mlp.experts = torch.nn.Module()
    layer.mlp.experts.local_experts = torch.nn.ModuleList()
    expected = {}
    for expert_index in expert_indices:
        expert = torch.nn.Module()
        for projection, key in (("linear_fc1", "gate_up"), ("linear_fc2", "down")):
            linear = torch.nn.Module()
            linear.input_quantizer = torch.nn.Module()
            scale = torch.tensor(float((expert_index + 1) * (2 if key == "down" else 1)))
            linear.input_quantizer._verl_input_global_scale = scale
            setattr(expert, projection, linear)
            expected[f"layers.4.mlp.experts.{expert_index}.{key}"] = scale
        layer.mlp.experts.local_experts.append(expert)
    chunk.decoder.layers = torch.nn.ModuleList([layer])
    scales = collect_nvfp4_input_global_scales([chunk])
    assert scales.keys() == expected.keys()
    config = build_compressed_tensors_config(linear=None, moe="NVFP4", ignore=[], config_file=None)
    quantizer = CompressedTensorsWeightQuantizer(config, input_global_scales=scales)
    for expert_index in expert_indices:
        for projection, key in (("gate_proj", "gate_up"), ("up_proj", "gate_up"), ("down_proj", "down")):
            prefix = f"model.layers.4.mlp.experts.{expert_index}.{projection}"
            result = dict(quantizer([(f"{prefix}.weight", torch.ones(32, 32))]))
            torch.testing.assert_close(
                result[f"{prefix}.input_global_scale"],
                expected[f"layers.4.mlp.experts.{expert_index}.{key}"].reshape(1),
                rtol=0,
                atol=0,
            )
