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

"""Exercise FP4 refit helpers without importing the optional vLLM package."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_fp4_utils():
    path = _REPO_ROOT / "verl/utils/vllm/vllm_fp4_utils.py"
    spec = importlib.util.spec_from_file_location("vllm_fp4_utils_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FP4_UTILS = _load_fp4_utils()
_MXFP4_LIVE_ATTR = _FP4_UTILS._MXFP4_LIVE_ATTR
_NVFP4_LIVE_ATTR = _FP4_UTILS._NVFP4_LIVE_ATTR
_restore_nvfp4_input_scales_in_place = _FP4_UTILS._restore_nvfp4_input_scales_in_place
_restore_nvfp4_kernel_config_in_place = _FP4_UTILS._restore_nvfp4_kernel_config_in_place
_stage_mxfp4_moe_params = _FP4_UTILS._stage_mxfp4_moe_params
_stage_nvfp4_moe_params = _FP4_UTILS._stage_nvfp4_moe_params


class _PackedMxfp4Method:
    _verl_uses_packed_checkpoint_names = True
    num_experts = 2
    intermediate_size = 32
    hidden_size = 32


def test_compressed_tensors_mxfp4_restage_exposes_packed_weight_names():
    module = torch.nn.Module()
    module.quant_method = _PackedMxfp4Method()
    module.weight_loader = lambda *args, **kwargs: None
    module.w13_weight = torch.nn.Parameter(torch.empty(2, 64, 16, dtype=torch.uint8), requires_grad=False)
    module.w2_weight = torch.nn.Parameter(torch.empty(2, 32, 16, dtype=torch.uint8), requires_grad=False)
    module.w13_weight_scale = torch.nn.Parameter(torch.empty(2, 64, 1, dtype=torch.uint8), requires_grad=False)
    module.w2_weight_scale = torch.nn.Parameter(torch.empty(2, 32, 1, dtype=torch.uint8), requires_grad=False)

    live_w13 = module.w13_weight
    live_w2 = module.w2_weight
    _stage_mxfp4_moe_params(module)

    assert module.w13_weight is live_w13
    assert module.w2_weight is live_w2
    assert module.w13_weight_packed.data_ptr() == live_w13.data_ptr()
    assert module.w2_weight_packed.data_ptr() == live_w2.data_ptr()
    assert module.w13_weight_packed.weight_loader == module.weight_loader
    assert getattr(module, _MXFP4_LIVE_ATTR)["w13_weight"] is live_w13
    assert getattr(module, _MXFP4_LIVE_ATTR)["w2_weight"] is live_w2
    assert module.w13_weight_scale.quant_method == "block"
    assert module.w2_weight_scale.quant_method == "block"


class _Nvfp4Moe:
    is_act_and_mul = True


class _Nvfp4Method:
    moe = _Nvfp4Moe()


def test_compressed_tensors_nvfp4_restage_exposes_checkpoint_layout():
    module = torch.nn.Module()
    module.quant_method = _Nvfp4Method()
    module.weight_loader = lambda *args, **kwargs: None
    module.num_experts = 2
    module.intermediate_size_per_partition = 32
    module.hidden_size = 32
    module.w13_weight = torch.nn.Parameter(torch.empty(2, 64, 16, dtype=torch.uint8), requires_grad=False)
    module.w2_weight = torch.nn.Parameter(torch.empty(2, 32, 16, dtype=torch.uint8), requires_grad=False)
    # CUTLASS swizzles/pads scales; use checkpoint shape here to exercise the
    # same-storage path without requiring CUDA.
    module.w13_weight_scale = torch.nn.Parameter(torch.empty(2, 64, 2, dtype=torch.float8_e4m3fn), requires_grad=False)
    module.w2_weight_scale = torch.nn.Parameter(torch.empty(2, 32, 2, dtype=torch.float8_e4m3fn), requires_grad=False)
    module.w13_weight_scale_2 = torch.nn.Parameter(torch.empty(2), requires_grad=False)
    module.w2_weight_scale_2 = torch.nn.Parameter(torch.empty(2), requires_grad=False)
    module.w13_weight_global_scale = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    module.w2_weight_global_scale = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    module.w13_input_global_scale = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    module.w2_input_global_scale = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    module.w13_input_scale = torch.ones(2)
    module.w2_input_scale = torch.ones(2)

    live_w13 = module.w13_weight
    live_global = module.w13_weight_global_scale
    live_input_scale = module.w13_input_scale
    _stage_nvfp4_moe_params(module)

    assert module.w13_weight_packed.data_ptr() == live_w13.data_ptr()
    assert module.w13_weight_global_scale is live_global
    assert module.w13_weight_scale.quant_method == "group"
    assert getattr(module, _NVFP4_LIVE_ATTR)["w13_weight"] is live_w13
    assert "w13_weight_scale_2" in getattr(module, _NVFP4_LIVE_ATTR)
    assert getattr(module, _NVFP4_LIVE_ATTR)["w13_input_scale"] is live_input_scale


def test_nvfp4_restage_updates_activation_scales_without_changing_addresses():
    module = torch.nn.Module()
    live_w13 = torch.zeros(2)
    live_w2 = torch.zeros(2)
    setattr(
        module,
        _NVFP4_LIVE_ATTR,
        {"w13_input_scale": live_w13, "w2_input_scale": live_w2},
    )
    converted = [None] * 8
    converted[3] = torch.tensor([2.0, 3.0])
    converted[7] = torch.tensor([4.0, 5.0])

    result = _restore_nvfp4_input_scales_in_place(module, converted)

    assert result[3] is live_w13
    assert result[7] is live_w2
    torch.testing.assert_close(live_w13, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(live_w2, torch.tensor([4.0, 5.0]))
    assert getattr(module, _NVFP4_LIVE_ATTR) == {}


def test_nvfp4_restage_retains_kernel_config_and_updates_derived_scales():
    module = torch.nn.Module()
    module.w13_weight_scale = torch.nn.Parameter(torch.ones(2))
    module.w2_weight_scale = torch.nn.Parameter(torch.ones(2))
    module.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(2))
    module.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(2))
    module.w13_input_scale = torch.tensor([0.5, 0.25])
    module.w2_input_scale = torch.tensor([0.2, 0.1])
    retained_config = SimpleNamespace(
        g1_alphas=module.w13_weight_scale_2,
        g2_alphas=module.w2_weight_scale_2,
        w1_scale=module.w13_weight_scale,
        w2_scale=module.w2_weight_scale,
        a1_gscale=torch.zeros(2),
        a2_gscale=torch.zeros(2),
    )
    retained_kernel = SimpleNamespace(fused_experts=SimpleNamespace(quant_config=retained_config))
    processing_config = SimpleNamespace(
        a1_gscale=1.0 / module.w13_input_scale,
        a2_gscale=1.0 / module.w2_input_scale,
    )
    module.quant_method = SimpleNamespace(moe_kernel=SimpleNamespace(), moe_quant_config=processing_config)

    _restore_nvfp4_kernel_config_in_place(module, retained_kernel, retained_config)

    assert module.quant_method.moe_kernel is retained_kernel
    assert module.quant_method.moe_quant_config is retained_config
    torch.testing.assert_close(retained_config.a1_gscale, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(retained_config.a2_gscale, torch.tensor([5.0, 10.0]))
