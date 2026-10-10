# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training-side configuration tests, not TE numerical-equivalence tests.

The engine method is extracted without importing Megatron providers, so its
guard wiring runs without building a model; real execution is covered by GPU
jobs.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl.utils.real_nvfp4.config import (
    real_nvfp4_quant_recipe_config,
    validate_real_nvfp4_model_contract,
    validate_real_nvfp4_parallelism,
    validate_real_nvfp4_te_recipe,
)

ENGINE_PATH = Path(__file__).resolve().parents[3] / "verl/workers/engine/megatron/transformer_impl.py"


def _engine_method(name):
    tree = ast.parse(ENGINE_PATH.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronEngine")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENGINE_PATH), "exec"), namespace)
    return namespace[name]


def _model_config(**kwargs):
    values = {"num_hidden_layers": 48, "num_experts": 128, "decoder_sparse_step": 1, "mlp_only_layers": []}
    values.update(kwargs)
    return SimpleNamespace(**values)


def _engine(start=2, end=4, pp=1, vpp=None):
    return SimpleNamespace(
        engine_config=SimpleNamespace(pipeline_model_parallel_size=pp, virtual_pipeline_model_parallel_size=vpp),
        _real_nvfp4_config=SimpleNamespace(num_layers_at_start_in_bf16=start, num_layers_at_end_in_bf16=end),
        model_config=SimpleNamespace(hf_config=_model_config()),
    )


def test_all_moe_model_is_supported():
    validate_real_nvfp4_model_contract(_model_config())


@pytest.mark.parametrize("overrides", [{"decoder_sparse_step": 2}, {"mlp_only_layers": [0, 1]}])
def test_mixed_dense_moe_is_rejected(overrides):
    with pytest.raises(ValueError, match="mixed dense/MoE"):
        validate_real_nvfp4_model_contract(_model_config(**overrides))


def test_shared_experts_are_rejected():
    with pytest.raises(ValueError, match="shared experts"):
        validate_real_nvfp4_model_contract(_model_config(shared_expert_intermediate_size=512))


@pytest.mark.parametrize("pp,vpp", [(2, None), (1, 2), (2, 2)])
def test_pipeline_splitting_is_rejected(pp, vpp):
    with pytest.raises(ValueError, match="pipeline_model_parallel_size=1"):
        validate_real_nvfp4_parallelism(pipeline_size=pp, virtual_pipeline_size=vpp)


def test_recipe_lists_carved_out_layers_before_general_patterns():
    matchers = real_nvfp4_quant_recipe_config(48, 2, 4)["matchers"]
    assert list(matchers) == [
        "layer_0_bf16",
        "layer_1_bf16",
        "layer_44_bf16",
        "layer_45_bf16",
        "layer_46_bf16",
        "layer_47_bf16",
        "attn_qkv_bf16",
        "attn_proj_bf16",
        "mlp_fc1_nvfp4",
        "mlp_fc2_nvfp4",
    ]
    assert matchers["layer_44_bf16"]["pattern"] == "*.layers.44.*"
    assert matchers["mlp_fc1_nvfp4"] == {"config": "nvfp4", "type": "glob", "pattern": "*.linear_fc1", "enabled": True}


@pytest.mark.parametrize("start,end", [(-1, 0), (0, -1), (24, 24)])
def test_recipe_rejects_invalid_carveout(start, end):
    with pytest.raises(ValueError, match="carve-out"):
        real_nvfp4_quant_recipe_config(48, start, end)


@pytest.mark.parametrize(
    "module_path,expected",
    [
        ("decoder.layers.0.mlp.experts.linear_fc1", "bf16"),
        ("decoder.layers.2.mlp.experts.linear_fc1", "nvfp4"),
        ("decoder.layers.43.mlp.experts.linear_fc2", "nvfp4"),
        ("decoder.layers.44.mlp.experts.linear_fc2", "bf16"),
        ("decoder.layers.10.self_attention.linear_qkv", "bf16"),
        ("decoder.layers.10.self_attention.linear_proj", "bf16"),
    ],
)
def test_engine_sets_megatron_recipe_and_fp4_overrides(module_path, expected):
    pytest.importorskip("megatron.core.quantization.quant_config")
    from megatron.core.quantization.utils import get_quant_config_or_none

    overrides = {"num_layers_at_start_in_bf16": 2}  # restating a value is allowed
    _engine_method("_apply_real_nvfp4_overrides")(_engine(), overrides)

    assert {key: value for key, value in overrides.items() if key != "quant_recipe"} == {
        "fp4": "e2m1",
        "fp4_recipe": "nvfp4",
        "fp4_param": False,
        "first_last_layers_bf16": True,
        "num_layers_at_start_in_bf16": 2,
        "num_layers_at_end_in_bf16": 4,
    }
    assert get_quant_config_or_none(module_path, overrides["quant_recipe"]).config_key == expected


def test_engine_without_carveout_disables_first_last_layers_bf16():
    pytest.importorskip("megatron.core.quantization.quant_config")
    overrides = {}
    _engine_method("_apply_real_nvfp4_overrides")(_engine(start=0, end=0), overrides)
    assert overrides["first_last_layers_bf16"] is False


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_layers_at_end_in_bf16": 2}, "num_layers_at_end_in_bf16=4"),
        ({"first_last_layers_bf16": False}, "first_last_layers_bf16=True"),
        ({"fp4_param": True}, "fp4_param=False"),
        ({"fp8": "e4m3"}, "FP8"),
        ({"quant_recipe": object()}, "owns"),
    ],
)
def test_engine_rejects_conflicting_overrides(overrides, match):
    pytest.importorskip("megatron.core.quantization.quant_config")
    with pytest.raises(ValueError, match=match):
        _engine_method("_apply_real_nvfp4_overrides")(_engine(), overrides)


def test_engine_rejects_pipeline_splitting():
    pytest.importorskip("megatron.core.quantization.quant_config")
    with pytest.raises(ValueError, match="pipeline_model_parallel_size=1"):
        _engine_method("_apply_real_nvfp4_overrides")(_engine(pp=2), {})


def _te_recipe(**overrides):
    qparams = SimpleNamespace(random_hadamard_transform=False, stochastic_rounding=False, fp4_2d_quantization=False)
    values = {
        "backward_override": "dequantized",
        "row_scaled_activation": True,
        "disable_rht": True,
        "disable_stochastic_rounding": True,
        "disable_2d_quantization": True,
        "nvfp4_4over6": "none",
        "nvfp4_4over6_e4m3_use_256": "all",
        "nvfp4_4over6_err_mode": "MAE",
        "fp4_quant_fwd_inp": qparams,
        "fp4_quant_fwd_weight": qparams,
        "fp4_quant_bwd_grad": qparams,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_te_recipe_contract_accepts_expected_semantics():
    validate_real_nvfp4_te_recipe(_te_recipe(), backward_override="dequantized")


@pytest.mark.parametrize(
    "overrides",
    [
        {"nvfp4_4over6": "all"},
        {"backward_override": "high_precision"},
        {"row_scaled_activation": False},
        {
            "fp4_quant_fwd_inp": SimpleNamespace(
                random_hadamard_transform=True, stochastic_rounding=False, fp4_2d_quantization=False
            )
        },
    ],
)
def test_te_recipe_contract_rejects_drift(overrides):
    with pytest.raises(RuntimeError, match="recipe drifted"):
        validate_real_nvfp4_te_recipe(_te_recipe(**overrides), backward_override="dequantized")
