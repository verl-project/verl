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
"""MXFP8 routed-expert token alignment on ROCm (apply_patch_rocm_mxfp8_expert_alignment)."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Load patch.py by path: importing the verl.models.mcore package pulls in megatron, which CI CPU hosts do not have.
_spec = importlib.util.spec_from_file_location(
    "mcore_patch_rocm_mxfp8", Path(__file__).parents[2] / "verl" / "models" / "mcore" / "patch.py"
)
_patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_patch)
ROCM_MXFP8_EXPERT_TOKEN_ALIGN = _patch.ROCM_MXFP8_EXPERT_TOKEN_ALIGN
apply_patch_rocm_mxfp8_expert_alignment = _patch.apply_patch_rocm_mxfp8_expert_alignment

# The parts of megatron.core.fp8_utils / moe_utils the patch relies on, as in Megatron-Core 0.19, for CI hosts without
# megatron: moe_utils imports get_fp8_align_size by name and get_align_size_for_quantization looks it up at call time.
_FP8_UTILS_SRC = """
def get_fp8_align_size(fp8_recipe):
    return 32 if fp8_recipe == "mxfp8" else 16
"""
_MOE_UTILS_SRC = """
from megatron.core.fp8_utils import get_fp8_align_size


def get_align_size_for_quantization(config):
    if config.use_transformer_engine_op_fuser:
        return 256
    if config.fp8:
        return get_fp8_align_size(config.fp8_recipe)
    if config.fp4:
        return 16
    return 0
"""


def _module(monkeypatch, name, src=""):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    exec(src, module.__dict__)
    return module


@pytest.fixture
def stub_megatron(monkeypatch):
    for name in ("megatron", "megatron.core", "megatron.core.transformer", "megatron.core.transformer.moe"):
        _module(monkeypatch, name)
    fp8_utils = _module(monkeypatch, "megatron.core.fp8_utils", _FP8_UTILS_SRC)
    moe_utils = _module(monkeypatch, "megatron.core.transformer.moe.moe_utils", _MOE_UTILS_SRC)
    sys.modules["megatron.core"].fp8_utils = fp8_utils
    sys.modules["megatron.core.transformer.moe"].moe_utils = moe_utils
    return SimpleNamespace(fp8_utils=fp8_utils, moe_utils=moe_utils)


def _config(fp8="e4m3", fp8_recipe="mxfp8", op_fuser=False):
    return SimpleNamespace(use_transformer_engine_op_fuser=op_fuser, fp8=fp8, fp8_recipe=fp8_recipe, fp4=None)


def test_rocm_pads_mxfp8_expert_tokens_to_256(monkeypatch, stub_megatron):
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    align = stub_megatron.moe_utils.get_align_size_for_quantization
    assert align(_config()) == 32

    apply_patch_rocm_mxfp8_expert_alignment()
    apply_patch_rocm_mxfp8_expert_alignment()  # idempotent: no second wrapper

    assert align(_config()) == ROCM_MXFP8_EXPERT_TOKEN_ALIGN == 256
    # enum-valued recipes (Megatron's Fp8Recipe) are matched by value
    assert align(_config(fp8_recipe=SimpleNamespace(value="mxfp8"))) == 256
    # other FP8 recipes, no FP8, and the TE op fuser keep Megatron's values
    assert align(_config(fp8_recipe="blockwise")) == 16
    assert align(_config(fp8=None)) == 0
    assert align(_config(op_fuser=True)) == 256
    # only the MoE token padding changes: fp8_utils' own function (GatedDeltaNet's width check) still says 32
    assert stub_megatron.fp8_utils.get_fp8_align_size("mxfp8") == 32
    wrapper = stub_megatron.moe_utils.get_fp8_align_size
    apply_patch_rocm_mxfp8_expert_alignment()
    assert stub_megatron.moe_utils.get_fp8_align_size is wrapper


def test_cuda_keeps_megatron_alignment(monkeypatch, stub_megatron):
    monkeypatch.setattr(torch.version, "hip", None)
    original = stub_megatron.moe_utils.get_fp8_align_size

    apply_patch_rocm_mxfp8_expert_alignment()

    assert stub_megatron.moe_utils.get_fp8_align_size is original
    assert stub_megatron.moe_utils.get_align_size_for_quantization(_config()) == 32


def test_real_megatron_on_rocm(monkeypatch):
    """Against the installed Megatron-Core, where available (skipped on CI hosts without megatron)."""
    moe_utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
    # restore the module attribute after the test
    monkeypatch.setattr(moe_utils, "get_fp8_align_size", moe_utils.get_fp8_align_size)
    monkeypatch.setattr(torch.version, "hip", "7.2.0")

    apply_patch_rocm_mxfp8_expert_alignment()

    assert moe_utils.get_align_size_for_quantization(_config()) == 256
    from megatron.core.fp8_utils import get_fp8_align_size

    assert get_fp8_align_size("mxfp8") == 32
