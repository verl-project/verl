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

The guards run on stand-in modules without building a model; real execution is
covered by GPU jobs.
"""

from types import SimpleNamespace

import pytest
import torch

from verl.utils.megatron.low_precision import bf16_layers_at_start_end
from verl.utils.real_nvfp4.config import (
    real_nvfp4_enabled,
    real_nvfp4_quant_recipe_config,
    validate_real_nvfp4_te_recipe,
)


def _model_config(**kwargs):
    values = {"num_hidden_layers": 48, "num_experts": 128}
    values.update(kwargs)
    return SimpleNamespace(**values)


def _fp4_overrides(**kwargs):
    return {
        "fp4": "e2m1",
        "fp4_recipe": "nvfp4",
        "first_last_layers_bf16": True,
        "num_layers_at_start_in_bf16": 2,
        "num_layers_at_end_in_bf16": 4,
        **kwargs,
    }


def _apply_overrides(overrides):
    from verl.utils.real_nvfp4.megatron import apply_real_nvfp4_overrides

    apply_real_nvfp4_overrides(overrides, _model_config(), pipeline_size=1, virtual_pipeline_size=None)


def _attest(root, tf_config=None):
    from verl.utils.real_nvfp4.megatron import attest_real_nvfp4_runtime

    attest_real_nvfp4_runtime(
        [root],
        tf_config or SimpleNamespace(fp4="e2m1", fp4_recipe="nvfp4"),
        SimpleNamespace(num_hidden_layers=6, n_routed_experts=8),
        1,
        1,
    )


def test_recipe_lists_carved_out_layers_before_general_patterns():
    matchers = real_nvfp4_quant_recipe_config(48, 2, 4)["matchers"]
    assert list(matchers) == [
        "layer_0_bf16",
        "layer_1_bf16",
        "layer_44_bf16",
        "layer_45_bf16",
        "layer_46_bf16",
        "layer_47_bf16",
        "experts_fc1_nvfp4",
        "experts_fc2_nvfp4",
        "other_bf16",
    ]
    assert matchers["layer_44_bf16"]["pattern"] == "*.layers.44.*"
    assert matchers["other_bf16"] == {"config": "bf16", "type": "glob", "pattern": "*", "enabled": True}


@pytest.mark.parametrize(
    "module_path,expected",
    [
        ("decoder.layers.0.mlp.experts.linear_fc1", "bf16"),
        ("decoder.layers.2.mlp.experts.linear_fc1", "nvfp4"),
        ("decoder.layers.43.mlp.experts.linear_fc2", "nvfp4"),
        ("decoder.layers.44.mlp.experts.linear_fc2", "bf16"),
        ("decoder.layers.10.self_attention.linear_qkv", "bf16"),
        ("decoder.layers.10.self_attention.linear_proj", "bf16"),
        ("decoder.layers.10.mlp.experts.local_experts.3.linear_fc1", "nvfp4"),
        # Dense MLPs, shared experts and other attention projections stay BF16, as in the rollout.
        ("decoder.layers.10.mlp.linear_fc1", "bf16"),
        ("decoder.layers.10.mlp.shared_experts.linear_fc2", "bf16"),
        ("decoder.layers.10.self_attention.linear_kv_down_proj", "bf16"),
        ("language_model.decoder.layers.10.self_attention.in_proj", "bf16"),
    ],
)
def test_fp4_override_limits_nvfp4_to_routed_experts(module_path, expected):
    pytest.importorskip("megatron.core.quantization.quant_config")
    from megatron.core.quantization.utils import get_quant_config_or_none

    overrides = _fp4_overrides()
    _apply_overrides(overrides)

    assert {key: value for key, value in overrides.items() if key != "quant_recipe"} == _fp4_overrides()
    assert get_quant_config_or_none(module_path, overrides["quant_recipe"]).config_key == expected


def test_nvfp4_training_is_switched_on_by_fp4():
    assert not real_nvfp4_enabled({})
    assert not real_nvfp4_enabled({"fp8": "e4m3"})
    assert real_nvfp4_enabled(_fp4_overrides())
    with pytest.raises(ValueError, match="fp4_recipe='nvfp4'"):
        real_nvfp4_enabled({"fp4": "e2m1", "fp4_recipe": "custom"})


def test_bf16_layers_follow_megatron_core_options():
    assert bf16_layers_at_start_end({}) == (0, 0)
    assert bf16_layers_at_start_end(_fp4_overrides()) == (2, 4)
    assert bf16_layers_at_start_end(_fp4_overrides(first_last_layers_bf16=False)) == (0, 0)
    # Megatron-Core keeps one layer at each end by default.
    assert bf16_layers_at_start_end({"fp8": "e4m3", "first_last_layers_bf16": True}) == (1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        bf16_layers_at_start_end({"first_last_layers_bf16": True, "num_layers_at_start_in_bf16": -1})


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"fp8": "e4m3"}, "combined with fp8"),
        ({"fp4_param": True}, "fp4_param=False"),
        ({"quant_recipe": {}}, "quant_recipe"),
    ],
)
def test_fp4_override_rejects_conflicting_options(extra, match):
    pytest.importorskip("megatron.core.quantization.quant_config")
    with pytest.raises(ValueError, match=match):
        _apply_overrides(_fp4_overrides(**extra))


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
    validate_real_nvfp4_te_recipe(_te_recipe())
    validate_real_nvfp4_te_recipe(_te_recipe(backward_override="high_precision"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"nvfp4_4over6": "all"},
        {"row_scaled_activation": False},
        # A worker started without NVTE_BACKWARD_OVERRIDE
        {"backward_override": None},
    ],
)
def test_te_recipe_contract_rejects_drift(overrides):
    with pytest.raises(RuntimeError, match="recipe drifted"):
        validate_real_nvfp4_te_recipe(_te_recipe(**overrides))


class NVFP4BlockScaling(SimpleNamespace):
    pass


def _deepseek_like_modules(recipe):
    """TE modules of a 6-layer model with a dense first layer, MLA and shared experts."""
    from megatron.core.quantization.utils import get_quant_config_or_none

    names = ["decoder.layers.0.mlp.linear_fc1", "decoder.layers.0.mlp.linear_fc2"]
    for layer in range(6):
        names += [f"decoder.layers.{layer}.self_attention.{p}" for p in ("linear_q_proj", "linear_kv_down_proj")]
    for layer in range(1, 6):
        names += [
            f"decoder.layers.{layer}.mlp.{part}.linear_fc{i}" for part in ("experts", "shared_experts") for i in (1, 2)
        ]
    root = torch.nn.Module()
    for name in names:
        module = root
        for part in name.split("."):
            if not hasattr(module, part):
                module.add_module(part, torch.nn.Module())
            module = getattr(module, part)
        key = get_quant_config_or_none(name, recipe).config_key
        fp4 = "nvfp4" if key == "nvfp4" else None
        training_recipe = SimpleNamespace(
            fp8_quantization_recipe=None, fp4_quantization_recipe=fp4, override_nonquantized_autocast=fp4 is not None
        )
        module.te_quant_params = SimpleNamespace(training_recipe=training_recipe)
    return root


def test_runtime_attestation_on_a_dense_mla_shared_expert_layout(monkeypatch):
    pytest.importorskip("megatron.core.quantization.quant_config")
    import megatron.core.fp4_utils
    from megatron.core.quantization.quant_config import RecipeConfig

    te_recipe = NVFP4BlockScaling(**vars(_te_recipe()))
    monkeypatch.setattr(megatron.core.fp4_utils, "get_fp4_recipe", lambda config: te_recipe)
    recipe = RecipeConfig.from_config_dict(real_nvfp4_quant_recipe_config(6, 1, 1))
    root = _deepseek_like_modules(recipe)
    _attest(root)

    # An expert that keeps NVFP4 only inside an FP4 autocast would run BF16 in a decoder
    # block that opens none, as Megatron-Bridge's Qwen3-VL block does.
    expert = root.get_submodule("decoder.layers.2.mlp.experts.linear_fc1")
    expert.te_quant_params.training_recipe.override_nonquantized_autocast = False
    with pytest.raises(RuntimeError, match="without autocast override"):
        _attest(root)
    expert.te_quant_params.training_recipe.override_nonquantized_autocast = True

    # A shared expert running in NVFP4 would not match the BF16 rollout.
    shared = root.get_submodule("decoder.layers.2.mlp.shared_experts.linear_fc1")
    shared.te_quant_params.training_recipe.fp4_quantization_recipe = "nvfp4"
    with pytest.raises(RuntimeError, match="unexpected precision"):
        _attest(root)


def test_runtime_attestation_accepts_a_vision_tower_built_without_fp4(monkeypatch):
    pytest.importorskip("megatron.core.quantization.quant_config")
    import megatron.core.fp4_utils
    from megatron.core.quantization.quant_config import RecipeConfig

    te_recipe = NVFP4BlockScaling(**vars(_te_recipe()))
    monkeypatch.setattr(megatron.core.fp4_utils, "get_fp4_recipe", lambda config: te_recipe)
    root = _deepseek_like_modules(RecipeConfig.from_config_dict(real_nvfp4_quant_recipe_config(6, 1, 1)))
    # A VLM vision tower gets no per-module params; its own config has no FP4, so it runs in BF16.
    vision = torch.nn.Module()
    vision.te_quant_params = None
    vision.config = SimpleNamespace(fp4=None)
    root.add_module("vision_model", torch.nn.Module())
    root.vision_model.add_module("linear_proj", vision)
    _attest(root)

    # The same module under an FP4 config would inherit the FP4 autocast.
    vision.config = SimpleNamespace(fp4="e2m1")
    with pytest.raises(RuntimeError, match="did not match vision_model.linear_proj"):
        _attest(root)


def test_module_init_reapplies_the_recipe_to_a_rebuilt_decoder(monkeypatch):
    pytest.importorskip("megatron.core.quantization.quant_config")
    import megatron.core.models.gpt
    from megatron.core.quantization.quant_config import RecipeConfig

    class FakeGPTModel(torch.nn.Module):
        pass

    class FakeTELinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.te_quant_params = None

        def finish_init(self, quantization_config):
            self.te_quant_params = quantization_config

    def add(root, name, module):
        *parents, leaf = name.split(".")
        for part in parents:
            if not hasattr(root, part):
                root.add_module(part, torch.nn.Module())
            root = getattr(root, part)
        root.add_module(leaf, module)

    monkeypatch.setattr(megatron.core.models.gpt, "GPTModel", FakeGPTModel)
    # A VLM language model that rebuilt its decoder after GPTModel.__init__, as
    # Megatron-Bridge's Qwen3-VL does: no decoder module has per-module params.
    language_model = FakeGPTModel()
    language_model.config = SimpleNamespace(
        quant_recipe=RecipeConfig.from_config_dict(real_nvfp4_quant_recipe_config(4, 1, 1))
    )
    expected = {
        "decoder.layers.0.mlp.experts.linear_fc1": "bf16",  # carved out
        "decoder.layers.1.mlp.experts.linear_fc1": "nvfp4",
        "decoder.layers.1.self_attention.in_proj": "bf16",
    }
    for name in expected:
        add(language_model, name, FakeTELinear())
    root = torch.nn.Module()
    root.add_module("language_model", language_model)
    vision = FakeTELinear()
    add(root, "vision_model.decoder.layers.0.linear_proj", vision)

    from verl.utils.real_nvfp4.megatron import finish_real_nvfp4_module_init

    finish_real_nvfp4_module_init([root])

    actual = {name: language_model.get_submodule(name).te_quant_params.config_key for name in expected}
    assert actual == expected
    # Outside every GPTModel, the module keeps its own (non-FP4) setup.
    assert vision.te_quant_params is None
