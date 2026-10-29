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

"""NVFP4_PER_TOKEN preset, trainer-side export and rollout refit checks, without vLLM."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytest.importorskip("compressed_tensors")

from verl.utils.qat.compressed_tensors import (  # noqa: E402
    NVFP4PerTokenWeightQuantizer,
    build_compressed_tensors_config,
    dispatch_rollout_weight_quantizer,
    nvfp4_per_token_module_matcher,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CARVE_IGNORE = [r"re:.*layers\.(0|1|6|7)\.mlp\.experts\..*"]
_HF = SimpleNamespace(num_hidden_layers=8, num_experts=2)


def _load_rollout_utils():
    # Load by path: the verl.utils.vllm package imports vLLM.
    path = _REPO_ROOT / "verl/utils/vllm/vllm_nvfp4_per_token_utils.py"
    spec = importlib.util.spec_from_file_location("vllm_nvfp4_per_token_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROLLOUT = _load_rollout_utils()


def _config(ignore=_CARVE_IGNORE):
    return build_compressed_tensors_config(linear=None, moe="NVFP4_PER_TOKEN", ignore=ignore, config_file=None)


def _dispatch(ignore=_CARVE_IGNORE, moe="NVFP4_PER_TOKEN", start=2, end=2, engine="vllm", hf_config=_HF):
    return dispatch_rollout_weight_quantizer(
        model_config=SimpleNamespace(hf_config=hf_config, local_path=None),
        rollout_config=SimpleNamespace(
            name=engine, quantization=SimpleNamespace(linear=None, moe=moe, ignore=ignore, config_file=None)
        ),
        real_nvfp4_bf16_layers=(start, end),
    )


def test_preset_is_nvfp4_weights_with_dynamic_inputs_and_moe_only():
    group = _config()["config_groups"]["group_moe"]
    assert group["weights"]["strategy"] == "tensor_group" and group["weights"]["group_size"] == 16
    assert group["input_activations"]["dynamic"] is True
    assert _ROLLOUT.is_nvfp4_per_token_scheme(group["weights"], group["input_activations"])
    static = build_compressed_tensors_config(linear=None, moe="NVFP4", ignore=[], config_file=None)
    group = static["config_groups"]["group_moe"]
    assert not _ROLLOUT.is_nvfp4_per_token_scheme(group["weights"], group["input_activations"])
    with pytest.raises(ValueError, match="only for routed MoE experts"):
        build_compressed_tensors_config(linear="NVFP4_PER_TOKEN", moe=None, ignore=[], config_file=None)


def test_per_token_matcher_follows_the_ignore_list_for_any_expert_naming():
    per_token = nvfp4_per_token_module_matcher(_config())
    assert per_token("model.layers.2.mlp.experts.0.gate_proj")
    assert per_token("language_model.model.layers.5.mlp.experts.3.down_proj")
    assert not per_token("model.layers.1.mlp.experts.0.gate_proj")
    assert not per_token("model.layers.2.mlp.shared_experts.gate_proj")
    assert not per_token("model.layers.2.self_attn.q_proj")


def _fake_quantize(weight, global_scale):
    rows, cols = weight.shape
    packed = torch.full((rows, cols // 2), int(global_scale.item()) % 256, dtype=torch.uint8)
    return packed, torch.ones((rows, cols // 16), dtype=torch.float8_e4m3fn)


def _export_experts(quantizer, layers, prefix="model.layers"):
    return list(quantizer([(f"{prefix}.{layer}.mlp.experts.gate_up_proj", torch.ones(2, 8, 32)) for layer in layers]))


@pytest.mark.parametrize(
    "hf_config",
    [
        # DeepSeek-V3 / GLM / Kimi style: leading dense layers and shared experts.
        SimpleNamespace(num_hidden_layers=8, n_routed_experts=4, first_k_dense_replace=1, n_shared_experts=1),
        # Qwen3.5 style: the decoder config is nested under text_config.
        SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=8, num_experts=4)),
    ],
)
def test_export_checks_the_carveout_on_other_layouts(monkeypatch, hf_config):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    _export_experts(_dispatch(hf_config=hf_config), range(8))
    # Training keeps layers 0, 1 and 7 in BF16, the rollout also layer 6.
    with pytest.raises(ValueError, match=r"layers\.6\.mlp\.experts\.gate_up_proj in BF16"):
        _export_experts(_dispatch(hf_config=hf_config, end=1), range(8))


def test_export_checks_the_carveout_under_the_exported_names(monkeypatch):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    # An ignore list anchored to a VLM's exported names, not to ``model.layers``.
    ignore = [r"re:model\.language_model\.layers\.(0|1|6|7)\.mlp\.experts\..*"]
    _export_experts(_dispatch(ignore=ignore), range(8), prefix="model.language_model.layers")
    with pytest.raises(ValueError, match=r"layers\.1\.mlp\.experts\.gate_up_proj in BF16"):
        _export_experts(_dispatch(ignore=ignore, start=1), range(8), prefix="model.language_model.layers")
    with pytest.raises(ValueError, match="in NVFP4"):
        _export_experts(_dispatch(ignore=ignore, start=2), range(8), prefix="model.layers")


def test_dispatch_pairs_real_nvfp4_with_the_preset():
    assert isinstance(_dispatch(), NVFP4PerTokenWeightQuantizer)
    with pytest.raises(ValueError, match="NVFP4_PER_TOKEN"):
        _dispatch(moe="NVFP4")
    with pytest.raises(ValueError, match="only with the vLLM rollout"):
        _dispatch(engine="sglang")


def test_exporter_shares_one_global_scale_across_gate_and_up(monkeypatch):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    prefix = "model.layers.2.mlp.experts.0"
    gate, up, down = torch.ones(4, 32), torch.full((4, 32), 4.0), torch.ones(32, 4 * 4)
    stream = [
        (f"{prefix}.gate_proj.weight", gate),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", gate),  # carved out: BF16 passthrough
        (f"{prefix}.down_proj.weight", down),
        (f"{prefix}.up_proj.weight", up),
    ]
    out = dict(NVFP4PerTokenWeightQuantizer(_config())(stream))
    assert torch.equal(out["model.layers.0.mlp.experts.0.gate_proj.weight"], gate)
    gate_scale, up_scale = out[f"{prefix}.gate_proj.weight_global_scale"], out[f"{prefix}.up_proj.weight_global_scale"]
    torch.testing.assert_close(gate_scale, torch.tensor([6.0 * 448.0 / 4.0]))
    assert torch.equal(gate_scale, up_scale)
    assert torch.equal(out[f"{prefix}.down_proj.input_global_scale"], torch.ones(1))
    assert out[f"{prefix}.up_proj.weight_packed"].shape == (4, 16)


def test_exporter_unpacks_only_quantized_packed_experts(monkeypatch):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    gate_up = torch.ones(2, 8, 32)
    stream = [(f"model.layers.{layer}.mlp.experts.gate_up_proj", gate_up) for layer in (0, 2)]
    out = dict(NVFP4PerTokenWeightQuantizer(_config())(stream))
    # The carved-out layer keeps its packed BF16 tensor; the quantized one is split per expert.
    assert torch.equal(out["model.layers.0.mlp.experts.gate_up_proj"], gate_up)
    assert {name for name in out if ".layers.2." in name} == {
        f"model.layers.2.mlp.experts.{expert}.{projection}.{suffix}"
        for expert in range(2)
        for projection in ("gate_proj", "up_proj")
        for suffix in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
    }


def test_exporter_rejects_an_unpaired_gate_proj(monkeypatch):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    with pytest.raises(RuntimeError, match="missing gate_proj/up_proj"):
        list(
            NVFP4PerTokenWeightQuantizer(_config())(
                [("model.layers.2.mlp.experts.0.gate_proj.weight", torch.ones(4, 32))]
            )
        )


_LAYER0_BF16 = [r"re:.*layers\.0\.mlp\.experts\..*"]


_HIDDEN, _INTERMEDIATE = 32, 16


def _expert_tensors(projection):
    rows, cols = (_HIDDEN, _INTERMEDIATE) if projection == "down_proj" else (_INTERMEDIATE, _HIDDEN)
    return {
        "weight": torch.ones(rows, cols, dtype=torch.bfloat16),
        "weight_packed": torch.ones(rows, cols // 2, dtype=torch.uint8),
        "weight_scale": torch.ones(rows, cols // 16).to(torch.float8_e4m3fn),
        "weight_global_scale": torch.ones(1),
        "input_global_scale": torch.ones(1),
    }


def _stream():
    weights = []
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            tensors = _expert_tensors(projection)
            weights.append((f"model.layers.0.mlp.experts.{expert}.{projection}.weight", tensors.pop("weight")))
            for suffix, tensor in tensors.items():
                weights.append((f"model.layers.1.mlp.experts.{expert}.{projection}.{suffix}", tensor))
    return weights


def _check(weights):
    check = _ROLLOUT.NVFP4PerTokenRefitCheck(_model([1], num_layers=2), _config(_LAYER0_BF16))
    # Split the round into two buckets, as the receiver delivers it.
    check.check_bucket(weights[: len(weights) // 2])
    check.check_bucket(weights[len(weights) // 2 :])
    check.finish()


def test_refit_check_accepts_a_complete_reordered_round():
    weights = list(reversed(_stream()))
    weights.insert(0, ("model.embed_tokens.weight", torch.ones(2, 16)))
    _check(weights)


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda w: w[:-1], "missing 1 expert tensors"),
        (lambda w: w + w[-1:], "duplicate"),
        (lambda w: [(n, t * 2 if n.endswith("input_global_scale") else t) for n, t in w], "scaled per token"),
        (
            lambda w: [(n, t * 2 if n.endswith("up_proj.weight_global_scale") else t) for n, t in w],
            "global scales differ",
        ),
        # The checkpoint layout of every expert tensor: shape, then dtype.
        (
            lambda w: [(n, t[:, :4] if n.endswith("0.down_proj.weight_packed") else t) for n, t in w],
            r"expected \(32, 8\)",
        ),
        (lambda w: [(n, t.float() if n.endswith("0.gate_proj.weight_scale") else t) for n, t in w], "float8_e4m3fn"),
        (lambda w: [(n, t.to(torch.uint8) if n.endswith(".0.up_proj.weight") else t) for n, t in w], "bf16"),
    ],
)
def test_refit_check_rejects_a_bad_round(corrupt, message):
    with pytest.raises(RuntimeError, match=message):
        _check(corrupt(_stream()))


def test_refit_check_accepts_packed_bf16_experts():
    # Packed-expert checkpoints (e.g. Qwen3.5) send their BF16 layers as one tensor per projection.
    weights = [(n, t) for n, t in _stream() if not n.startswith("model.layers.0.")]
    gate_up = torch.ones(2, 2 * _INTERMEDIATE, _HIDDEN, dtype=torch.bfloat16)
    down = torch.ones(2, _HIDDEN, _INTERMEDIATE, dtype=torch.bfloat16)
    prefix = "model.layers.0.mlp.experts"
    _check(weights + [(f"{prefix}.gate_up_proj", gate_up), (f"{prefix}.down_proj", down)])
    with pytest.raises(RuntimeError, match="expected"):
        _check(weights + [(f"{prefix}.gate_up_proj", down), (f"{prefix}.down_proj", down)])


def test_worker_reads_the_quantization_config_vllm_built_its_methods_from():
    per_token = _config()
    static = build_compressed_tensors_config(linear=None, moe="NVFP4", ignore=[], config_file=None)

    def vllm_config(built, hf):
        return SimpleNamespace(
            quant_config=None if built is None else SimpleNamespace(config=built),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(quantization_config=hf)),
        )

    find = _ROLLOUT.nvfp4_per_token_quantization_config
    # vLLM's built config wins over the HF config, which is only the fallback.
    assert find(vllm_config(per_token, None)) == per_token
    assert find(vllm_config(static, per_token)) is None
    assert find(vllm_config(None, per_token)) == per_token


class CompressedTensorsW4A4Nvfp4PerTokenMoEMethod:
    def __init__(self, per_token_activation: bool = True) -> None:
        self.moe_kernel = SimpleNamespace(fused_experts=SimpleNamespace(per_token_activation=per_token_activation))


class UnquantizedFusedMoEMethod:
    pass


_MOE_CONFIG = SimpleNamespace(
    hidden_dim_unpadded=_HIDDEN,
    intermediate_size_per_partition_unpadded=_INTERMEDIATE,
    moe_parallel_config=SimpleNamespace(tp_size=1),
)


class _FakeNVFP4MoE(torch.nn.Module):
    def __init__(self, per_token_activation: bool = True) -> None:
        super().__init__()
        self.quant_method = CompressedTensorsW4A4Nvfp4PerTokenMoEMethod(per_token_activation)
        self.global_num_experts = 2
        self.moe_config = _MOE_CONFIG
        self._already_called_process_weights_after_loading = True
        self.w13_weight_scale = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        self.w2_weight_scale = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        self.w13_weight_scale_2 = torch.ones(2)
        self.w2_weight_scale_2 = torch.ones(2)
        self.w13_input_scale = torch.ones(2)
        self.w2_input_scale = torch.ones(2)
        self.g1_scale_c = self.w13_weight_scale_2.clone()
        experts = self.quant_method.moe_kernel.fused_experts
        experts.quant_config = SimpleNamespace(
            g1_alphas=self.w13_weight_scale_2,
            g2_alphas=self.w2_weight_scale_2,
            w1_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
            a1_gscale=torch.ones(2),
            a2_gscale=torch.ones(2),
        )
        self.quant_method.moe_quant_config = experts.quant_config
        experts.moe_config = SimpleNamespace(is_act_and_mul=True)
        experts.g1_scale_c = self.g1_scale_c


class _FakeUnquantizedMoE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant_method = UnquantizedFusedMoEMethod()
        self.global_num_experts = 2
        self.moe_config = _MOE_CONFIG


def _model(quantized_layers, num_layers=4, **kwargs):
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList()
    for index in range(num_layers):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Module()
        quantized = index in quantized_layers
        layer.mlp.experts.routed_experts = _FakeNVFP4MoE(**kwargs) if quantized else _FakeUnquantizedMoE()
        root.model.layers.append(layer)
    return root


_LAYERS_0_3_BF16 = _config([r"re:.*layers\.(0|3)\.mlp\.experts\..*"])


@pytest.fixture
def vllm_method_types(monkeypatch):
    """Serve the fake method classes from the vLLM modules the attestation imports."""
    ct_moe = ModuleType("compressed_tensors_moe")
    ct_moe.CompressedTensorsMoEMethod = SimpleNamespace(
        _verl_nvfp4_per_token_method=CompressedTensorsW4A4Nvfp4PerTokenMoEMethod
    )
    unquantized = ModuleType("unquantized_fused_moe_method")
    unquantized.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod
    layers = "vllm.model_executor.layers"
    monkeypatch.setitem(sys.modules, f"{layers}.quantization.compressed_tensors.compressed_tensors_moe", ct_moe)
    monkeypatch.setitem(sys.modules, f"{layers}.fused_moe.unquantized_fused_moe_method", unquantized)


@pytest.mark.usefixtures("vllm_method_types")
def test_runtime_attestation_checks_layer_precision_and_per_token_kernel():
    _ROLLOUT.attest_nvfp4_per_token_runtime(_model([1, 2]), _LAYERS_0_3_BF16)
    with pytest.raises(RuntimeError, match="wrong routed-expert precision"):
        _ROLLOUT.attest_nvfp4_per_token_runtime(_model([1]), _LAYERS_0_3_BF16)
    with pytest.raises(RuntimeError, match="wrong routed-expert precision"):
        _ROLLOUT.attest_nvfp4_per_token_runtime(_model([0, 1, 2]), _LAYERS_0_3_BF16)
    with pytest.raises(RuntimeError, match="per token"):
        _ROLLOUT.attest_nvfp4_per_token_runtime(_model([1, 2], per_token_activation=False), _LAYERS_0_3_BF16)


@pytest.mark.usefixtures("vllm_method_types")
def test_runtime_attestation_rejects_a_stale_derived_scale():
    model = _model([1, 2])
    model.model.layers[1].mlp.experts.routed_experts.w13_weight_scale_2.mul_(2)
    with pytest.raises(RuntimeError, match="g1_scale_c does not match"):
        _ROLLOUT.attest_nvfp4_per_token_runtime(model, _LAYERS_0_3_BF16)


def test_per_token_rollout_check_reads_a_config_file(tmp_path):
    pytest.importorskip("vllm")
    import json

    from verl.utils.qat.compressed_tensors import is_nvfp4_per_token_config
    from verl.utils.real_nvfp4 import check_nvfp4_per_token_rollout
    from verl.workers.config.rollout import RolloutQuantizationConfig

    config_file = tmp_path / "quantization_config.json"
    config_file.write_text(json.dumps(_config()))
    built = RolloutQuantizationConfig(config_file=str(config_file)).build_quantization_config(None)
    assert is_nvfp4_per_token_config(built)
    # A checkpoint that ships the config is detected the same way.
    assert is_nvfp4_per_token_config(SimpleNamespace(to_dict=lambda: built))
    static = build_compressed_tensors_config(linear=None, moe="NVFP4", ignore=[], config_file=None)
    assert not is_nvfp4_per_token_config(static)

    rollout = SimpleNamespace(
        tensor_model_parallel_size=4,
        expert_parallel_size=4,
        data_parallel_size=1,
        mtp=None,
        checkpoint_engine=SimpleNamespace(backend="naive"),
    )

    def check(compilation_config, engine_kwargs, config=built):
        check_nvfp4_per_token_rollout(rollout, config, compilation_config, engine_kwargs)

    # Other quantization configs are left alone.
    check_nvfp4_per_token_rollout(rollout, static, {"cudagraph_mode": "FULL_AND_PIECEWISE"}, {"enable_eplb": True})
    # Expert parallelism within one engine keeps whole experts per rank.
    compilation_config = {}
    check(compilation_config, {})
    assert compilation_config == {"cudagraph_mode": "FULL_DECODE_ONLY"}
    with pytest.raises(ValueError, match="FULL_DECODE_ONLY"):
        check({"cudagraph_mode": "FULL_AND_PIECEWISE"}, {})
    with pytest.raises(ValueError, match="no EPLB"):
        check({}, {"enable_eplb": True})
    for context_parallel in ("decode_context_parallel_size", "prefill_context_parallel_size"):
        with pytest.raises(ValueError, match="context parallelism"):
            check({}, {context_parallel: 2})
    # Tensor-parallel experts
    rollout.expert_parallel_size = 1
    check({}, {})
    # Data-parallel attention with all2all expert parallelism
    rollout.tensor_model_parallel_size, rollout.data_parallel_size = 1, 4
    with pytest.raises(ValueError, match="data_parallel_size=1"):
        check({}, {})
