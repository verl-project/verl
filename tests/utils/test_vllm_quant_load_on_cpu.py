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

"""CPU regressions for FP8 weight/scale pairing during a LoRA base sync."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

_VLLM_DIR = Path(__file__).resolve().parents[2] / "verl/utils/vllm"


@pytest.fixture
def quant_loader(monkeypatch):
    """Load the real quant utilities without importing vLLM's package initializer."""
    for name in ("vllm_fp4_utils", "vllm_fp8_utils", "vllm_quant_utils"):
        qualified_name = f"verl.utils.vllm.{name}"
        spec = importlib.util.spec_from_file_location(qualified_name, _VLLM_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified_name, module)
        spec.loader.exec_module(module)

    # Name resolution is independent of pairing; these tests use live names.
    utils = types.ModuleType("verl.utils.vllm.utils")
    utils.resolve_weight_name = lambda model, name, *args, **kwargs: name
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return module.load_quanted_weights


def _runner():
    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.k_proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.load_weights = Mock(side_effect=lambda weights: {name for name, _ in weights})
    return types.SimpleNamespace(model=model, drafter=types.SimpleNamespace(model=model))


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("is_drafter", [False, True])
def test_orphaned_fp8_weights_raise_before_loading(quant_loader, dtype, is_drafter):
    """Identify every unpaired weight and avoid applying a partial bucket."""
    runner = _runner()
    weights = [(name, torch.ones(2, 2).to(dtype)) for name in ("q_proj.weight", "k_proj.weight")]

    with pytest.raises(RuntimeError, match="FP8 weights missing weight_scale or weight_scale_inv") as exc:
        quant_loader(iter(weights), runner, is_drafter=is_drafter, peft_config={"r": 1})

    for name, _ in weights:
        assert name in str(exc.value)
    runner.model.load_weights.assert_not_called()


@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
@pytest.mark.parametrize("scale_first", [False, True])
def test_complete_fp8_pairs_are_dequantized(quant_loader, scale_name, scale_first):
    """Accept either scale spelling and arrival order, including generator inputs."""
    runner = _runner()
    weights = [
        ("q_proj.weight", torch.ones(2, 2).to(torch.float8_e4m3fn)),
        (f"q_proj.{scale_name}", torch.full((1, 1), 3.0)),
    ]
    if scale_first:
        weights.reverse()

    loaded = quant_loader(iter(weights), runner, peft_config={"r": 1})

    assert loaded == {"q_proj.weight"}
    [(name, weight)] = runner.model.load_weights.call_args.args[0]
    assert name == "q_proj.weight"
    torch.testing.assert_close(weight, torch.full((2, 2), 3.0, dtype=torch.bfloat16))


@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("weight_suffix, scale_suffix", [(".base_layer", ""), ("", ".base_layer")])
def test_fused_fp8_pairs_ignore_base_layer_for_pairing(
    quant_loader, scale_name, scale_first, weight_suffix, scale_suffix
):
    """Pair checkpoint shards of a fused BF16 owner while keeping the loader's weight name."""
    runner = _runner()
    runner.model.attn = torch.nn.Module()
    runner.model.attn.qkv_proj = torch.nn.Linear(2, 6, bias=False, dtype=torch.bfloat16)
    runner.model.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    weight_name = f"attn.q_proj{weight_suffix}.weight"
    weights = [
        (weight_name, torch.ones(2, 2).to(torch.float8_e4m3fn)),
        (f"attn.q_proj{scale_suffix}.{scale_name}", torch.full((1, 1), 3.0)),
    ]
    if scale_first:
        weights.reverse()

    loaded = quant_loader(iter(weights), runner, peft_config={"r": 1})

    assert loaded == {weight_name}
    [(name, weight)] = runner.model.load_weights.call_args.args[0]
    assert name == weight_name
    torch.testing.assert_close(weight, torch.full((2, 2), 3.0, dtype=torch.bfloat16))


@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
def test_scale_only_entries_are_ignored(quant_loader, scale_name):
    """Trainer-only scales for BF16 parameters must remain harmless."""
    runner = _runner()
    weight = torch.ones(2, 2, dtype=torch.bfloat16)

    loaded = quant_loader(
        [("q_proj.weight", weight), (f"k_proj.{scale_name}", torch.ones(1, 1))],
        runner,
        peft_config={"r": 1},
    )

    assert loaded == {"q_proj.weight"}
    [(name, loaded_weight)] = runner.model.load_weights.call_args.args[0]
    assert name == "q_proj.weight"
    assert loaded_weight is weight
