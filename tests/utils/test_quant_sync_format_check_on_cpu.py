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
"""CPU tests for the export-vs-engine format check and its vLLM name resolution (vLLM modules faked)."""

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.utils.quant_sync_format_check import SyncFormatCheck

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_by_path(name, relpath):
    # verl.utils.vllm's package __init__ imports vLLM itself; this module needs only torch.
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vllm = _load_by_path("vllm_sync_format_check_under_test", "verl/utils/vllm/vllm_sync_format_check.py")
resolve_vllm_param, vllm_sync_format_check = _vllm.resolve_vllm_param, _vllm.vllm_sync_format_check

FP8, BF16, U8 = torch.float8_e4m3fn, torch.bfloat16, torch.uint8


@pytest.fixture(autouse=True)
def _audit_logs_reach_caplog(caplog, monkeypatch):
    # The audit logger does not propagate (it writes to stderr on its own); attach caplog's handler to it.
    monkeypatch.delenv("VERL_QUANT_LAYER_AUDIT", raising=False)
    log = logging.getLogger("verl.utils.quant_layer_audit")
    log.addHandler(caplog.handler)
    level = log.level
    log.setLevel(logging.INFO)
    yield
    log.setLevel(level)
    log.removeHandler(caplog.handler)


def _t(dtype, *shape):
    return torch.zeros(*(shape or (4, 4)), dtype=dtype)


def _param(dtype, *shape):
    return torch.nn.Parameter(_t(dtype, *shape), requires_grad=False)


# ---------------------------------------------------------------------------
# The check itself, on an engine whose parameters are looked up by name
# ---------------------------------------------------------------------------


def _engine(**params):
    model = torch.nn.Module()
    model.params = {name.replace("__", "."): _param(dtype) for name, dtype in params.items()}
    return model


def _check(model):
    return SyncFormatCheck.of(model, "test", lambda m, name: m.params.get(name))


def test_the_first_sync_is_checked_bucket_by_bucket_and_later_syncs_are_not(caplog):
    model = _engine(a__weight=FP8, b__weight=BF16)
    check = _check(model)
    check.begin_sync()
    check.check_bucket(model, [("a.weight", _t(FP8)), ("a.weight_scale", _t(U8, 4, 1))])
    assert (check.checked, check.disagreeing) == (1, 0) and "disagree" not in caplog.text
    check.check_bucket(model, [("b.weight", _t(FP8))])  # the export quantized a layer the engine keeps in BF16
    assert check.disagreeing == 1
    assert "b.weight: the sync sends float8_e4m3fn, the engine parameter is bfloat16" in caplog.text
    assert "The sync goes on" in caplog.text
    check.end_sync()
    assert check.finished and "first weight sync: 2 weights checked, 1 in a format other" in caplog.text
    caplog.clear()
    check.begin_sync()  # the next sync sends the same formats: nothing is read or reported
    check.check_bucket(model, [("b.weight", _t(FP8))])
    check.end_sync()
    assert (check.checked, check.disagreeing) == (2, 1) and caplog.text == ""


def test_raise_mode_stops_the_sync_before_the_bucket_is_written(monkeypatch):
    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT", "raise")
    model = _engine(a__weight=BF16)
    with pytest.raises(RuntimeError, match="stopped before this bucket is written") as err:
        _check(model).check_bucket(model, [("a.weight", _t(FP8))])
    assert "a.weight: the sync sends float8_e4m3fn, the engine parameter is bfloat16" in str(err.value)


def test_scales_are_not_judged_and_unmapped_names_are_counted():
    model = _engine(a__weight=FP8)
    check = _check(model)
    check.check_bucket(
        model,
        [
            ("a.weight", _t(FP8)),
            ("a.weight_scale", _t(U8)),  # a scale, not a weight
            ("a.input_scale", _t(torch.float32)),
            ("visual.qkv.weight", _t(BF16)),  # no such engine parameter
        ],
    )
    assert (check.checked, check.disagreeing, check.unmapped) == (1, 0, 1)


def test_packed_fp4_counts_as_quantized():
    model = _engine(a__weight_packed=U8, b__weight=BF16)
    check = _check(model)
    model.params["b.weight_packed"] = model.params["b.weight"]  # a layer the engine kept in BF16
    check.check_bucket(model, [("a.weight_packed", _t(U8)), ("b.weight_packed", _t(U8))])
    assert (check.checked, check.disagreeing) == (2, 1)


def test_a_sync_that_does_not_finish_is_checked_again_in_full():
    model = _engine(a__weight=FP8, b__weight=FP8)
    check = _check(model)
    check.begin_sync()
    check.check_bucket(model, [("a.weight", _t(FP8))])  # the sync fails after its first bucket
    check.begin_sync()
    check.check_bucket(model, [("a.weight", _t(FP8))])
    check.check_bucket(model, [("b.weight", _t(FP8))])
    check.end_sync()
    assert check.finished and check.checked == 2  # one full sync, not the failed one added on top


def test_the_check_can_be_disabled_and_a_sync_without_buckets_does_not_finish_it(monkeypatch):
    model = _engine(a__weight=BF16)
    check = _check(model)
    check.begin_sync()
    check.end_sync()  # e.g. a finalize-only request
    assert not check.finished
    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT", "0")
    check.check_bucket(model, [("a.weight", _t(FP8))])
    assert check.checked == 0


def test_the_check_lives_on_the_model():
    model = _engine()
    assert _check(model) is _check(model)
    assert _check(_engine()) is not _check(model)


# ---------------------------------------------------------------------------
# vLLM: the parameter that receives each HF weight
# ---------------------------------------------------------------------------


class PPMissingLayer(torch.nn.Module):
    pass


def _linear(dtype, packed=False):
    module = torch.nn.Module()
    if packed:
        module.weight_packed = _param(U8)
    else:
        module.weight = _param(dtype)
    return module


def _vllm_model():
    """A vLLM-shaped Qwen-MoE-ish model: fused qkv / gate_up, a MoE runner, a LoRA wrapper, a PP stub."""
    model = torch.nn.Module()
    model.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.qkv_proj = _linear(FP8)
    lora = torch.nn.Module()
    lora.base_layer = _linear(FP8)
    layer.self_attn.o_proj = lora
    layer.mlp = torch.nn.Module()
    layer.mlp.gate = _linear(BF16)  # the router, kept in BF16
    layer.mlp.gate_up_proj = _linear(FP8)
    layer.mlp.down_proj = _linear(None, packed=True)  # NVFP4: compressed-tensors' packed weight
    runner = torch.nn.Module()
    runner.routed_experts = torch.nn.Module()
    runner.routed_experts.w13_weight = _param(FP8, 2, 8, 4)
    runner.routed_experts.w2_weight = _param(FP8, 2, 4, 4)
    layer.mlp.experts = runner
    model.model.layers = torch.nn.ModuleList([layer, PPMissingLayer()])
    model.model.norm = _linear(BF16)
    model.lm_head = _linear(BF16)
    return model


def test_resolve_vllm_param_finds_the_receiving_parameter():
    model = _vllm_model()
    layer = model.model.layers[0]
    experts = layer.mlp.experts.routed_experts
    expected = {
        "model.layers.0.self_attn.k_proj.weight": layer.self_attn.qkv_proj.weight,  # packed into qkv_proj
        "model.layers.0.self_attn.o_proj.weight": layer.self_attn.o_proj.base_layer.weight,  # under a LoRA wrapper
        "model.layers.0.mlp.gate.weight": layer.mlp.gate.weight,
        "model.layers.0.mlp.up_proj.weight": layer.mlp.gate_up_proj.weight,
        "model.layers.0.mlp.down_proj.weight_packed": layer.mlp.down_proj.weight_packed,
        "model.layers.0.mlp.gate.weight_packed": layer.mlp.gate.weight,  # packed weight sent to a BF16 layer
        "model.layers.0.mlp.experts.3.gate_proj.weight": experts.w13_weight,
        "model.layers.0.mlp.experts.3.down_proj.weight": experts.w2_weight,
        "model.layers.0.mlp.experts.gate_up_proj": experts.w13_weight,  # transformers >= 5 fused names
        "model.layers.0.mlp.experts.down_proj": experts.w2_weight,
        "model.norm.weight": model.model.norm.weight,
        "lm_head.weight": model.lm_head.weight,
    }
    for name, param in expected.items():
        assert resolve_vllm_param(model, name) is param, name
    assert resolve_vllm_param(model, "model.layers.1.self_attn.q_proj.weight") is None  # another pipeline stage
    assert resolve_vllm_param(model, "model.visual.blocks.0.attn.qkv.weight") is None  # no such module


def test_resolve_vllm_param_applies_the_hf_to_vllm_mapper():
    model = _vllm_model()

    class _Mapper:
        def apply_list(self, names):
            return [n.replace("model.language_model.", "model.") for n in names if not n.startswith("drop.")]

    model.hf_to_vllm_mapper = _Mapper()
    layer = model.model.layers[0]
    assert resolve_vllm_param(model, "model.language_model.layers.0.mlp.gate.weight") is layer.mlp.gate.weight
    assert resolve_vllm_param(model, "drop.me.weight") is None


def test_vllm_check_catches_an_engine_that_kept_gate_up_in_bf16(caplog):
    # vLLM 0.24's ModelOpt matcher read the router exclusion mlp.gate as a prefix of mlp.gate_up_proj
    model = _vllm_model()
    model.model.layers[0].mlp.gate_up_proj.weight = _param(BF16)
    check = vllm_sync_format_check(model, SimpleNamespace())
    check.begin_sync()
    check.check_bucket(model, [(f"model.layers.0.mlp.{p}.weight", _t(FP8)) for p in ("gate_proj", "up_proj")])
    assert check.disagreeing == 2 and "the vllm engine disagree on the format of 2 weight(s)" in caplog.text


def test_vllm_check_is_skipped_when_the_engine_quantizes_what_it_receives():
    model = _vllm_model()
    online = SimpleNamespace(is_checkpoint_fp8_serialized=False)  # an unquantized checkpoint, FP8 at load
    assert vllm_sync_format_check(model, online) is None
    serialized = vllm_sync_format_check(model, SimpleNamespace(is_checkpoint_fp8_serialized=True))
    assert serialized is vllm_sync_format_check(model, None)  # compressed-tensors has no such flag
