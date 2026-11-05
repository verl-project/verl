# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""ModelOpt fake quantization and Megatron QAT workflow helpers."""

import copy

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    GroupedQuantizer,
    TensorQuantizer,
    is_registered_quant_backend,
    register_quant_backend,
)

_FP8_BLOCK_BACKEND = "verl_fp8_block"
_MXFP8_BACKEND = "verl_mxfp8"
_MXFP4_BACKEND = "verl_mxfp4"
_NVFP4_WEIGHT_BACKEND = "verl_nvfp4_weight"
_NVFP4_INPUT_BACKEND = "verl_nvfp4_input"


class _Fp8FakeQuant(torch.autograd.Function):
    """TileLang FP8 fake quantization over blocks of the last two dimensions."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        block_rows: int,
        block_cols: int,
        scale_fmt: str | None,
    ) -> torch.Tensor:
        from verl.utils.qat.fp8 import fp8_quant

        rows, cols = x.shape[-2:]
        pad_rows, pad_cols = -rows % block_rows, -cols % block_cols
        padded = F.pad(x, (0, pad_cols, 0, pad_rows)) if pad_rows or pad_cols else x
        dq = fp8_quant(padded.detach(), (block_rows, block_cols), scale_fmt=scale_fmt, dequant=True)
        return dq[..., :rows, :cols]

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


def _fp8_block_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    block_rows, block_cols = quantizer.backend_extra_args["block_sizes"]
    return _Fp8FakeQuant.apply(inputs, block_rows, block_cols, None)


def _mxfp8_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    return _Fp8FakeQuant.apply(inputs, 1, 32, "e8m0")


class _Mxfp4FakeQuant(torch.autograd.Function):
    """TileLang MXFP4 fake quantization with STE."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        from verl.utils.qat.fp4 import fp4_quant

        return fp4_quant(x.detach(), dequant=True)

    @staticmethod
    def backward(ctx, grad):
        return grad


def _mxfp4_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    return _Mxfp4FakeQuant.apply(inputs)


class _GroupedFp8FakeQuant(torch.autograd.Function):
    """Pointer-table FP8 QDQ with an STE back to separate expert parameters."""

    @staticmethod
    def forward(ctx, block_rows: int, block_cols: int, scale_fmt: str | None, *weights: torch.Tensor):
        from verl.utils.qat.fp8 import fp8_grouped_fake_quant

        return fp8_grouped_fake_quant(weights, (block_rows, block_cols), scale_fmt=scale_fmt)

    @staticmethod
    def backward(ctx, grad):
        return None, None, None, *grad.unbind(0)


class _GroupedMxfp4FakeQuant(torch.autograd.Function):
    """Pointer-table MXFP4 QDQ with an STE back to separate expert parameters."""

    @staticmethod
    def forward(ctx, *weights: torch.Tensor):
        from verl.utils.qat.fp4 import fp4_grouped_fake_quant

        return fp4_grouped_fake_quant(weights)

    @staticmethod
    def backward(ctx, grad):
        return tuple(grad.unbind(0))


class _Nvfp4FakeQuant(torch.autograd.Function):
    """NVFP4 QDQ of the rollout's quantization, with straight-through grad."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, per_token: bool) -> torch.Tensor:
        from verl.utils.qat.nvfp4 import fake_quantize_nvfp4, fake_quantize_nvfp4_per_token, nvfp4_global_scale

        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        if per_token:
            return fake_quantize_nvfp4_per_token(x2d).reshape(shape)
        return fake_quantize_nvfp4(x2d, nvfp4_global_scale(x2d)).reshape(shape)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _nvfp4_weight_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    # One global scale per expert matrix, as the export computes it; the fused linear_fc1
    # holds gate_proj and up_proj, which the export gives one shared scale.
    if inputs.ndim > 2:
        # TEGroupedLinear stores one expert matrix per leading entry.
        return torch.stack([_Nvfp4FakeQuant.apply(weight, False) for weight in inputs], dim=0)
    return _Nvfp4FakeQuant.apply(inputs, False)


def _nvfp4_input_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    # The rollout's per-token MoE kernel quantizes each token with its own global scale.
    return _Nvfp4FakeQuant.apply(inputs, True)


if not is_registered_quant_backend(_FP8_BLOCK_BACKEND):
    register_quant_backend(_FP8_BLOCK_BACKEND, _fp8_block_backend)
if not is_registered_quant_backend(_MXFP8_BACKEND):
    register_quant_backend(_MXFP8_BACKEND, _mxfp8_backend)
if not is_registered_quant_backend(_MXFP4_BACKEND):
    register_quant_backend(_MXFP4_BACKEND, _mxfp4_backend)
if not is_registered_quant_backend(_NVFP4_WEIGHT_BACKEND):
    register_quant_backend(_NVFP4_WEIGHT_BACKEND, _nvfp4_weight_backend)
if not is_registered_quant_backend(_NVFP4_INPUT_BACKEND):
    register_quant_backend(_NVFP4_INPUT_BACKEND, _nvfp4_input_backend)

# Mirrors the compressed-tensors "fp8" rollout scheme: 128x128 weight blocks and
# dynamic 1x128 activation groups, both with fp32 scales. Only linear GEMMs the
# rollout quantizes may be enabled; anything else (attention BMMs, GDN conv1d,
# vision tower) would make training see a different model than the rollout.
_FP8_BLOCK_QUANTIZER_CFG = {
    "*": {"enable": False},
    "*weight_quantizer": {
        "num_bits": (4, 3),
        "backend": _FP8_BLOCK_BACKEND,
        "backend_extra_args": {"block_sizes": [128, 128]},
        "enable": True,
    },
    "*input_quantizer": {
        "num_bits": (4, 3),
        "backend": _FP8_BLOCK_BACKEND,
        "backend_extra_args": {"block_sizes": [1, 128]},
        "enable": True,
    },
    "*conv1d*": {"enable": False},
    "*vision_model*": {"enable": False},
}

_QUANTIZER_CFGS = {"fp8": _FP8_BLOCK_QUANTIZER_CFG}
_SUPPORTED_QAT_MODES = {*_QUANTIZER_CFGS, "mxfp8", "mxfp4", "mxfp4_mxfp8", "nvfp4"}

_ROUTED_EXPERT_WEIGHT_QUANTIZER = "*mlp.experts*weight_quantizer"
_ROUTED_EXPERT_INPUT_QUANTIZER = "*mlp.experts*input_quantizer"
_GROUPABLE_WEIGHT_BACKENDS = {_FP8_BLOCK_BACKEND, _MXFP8_BACKEND, _MXFP4_BACKEND}


def _modelopt_quantizer_cfg(preset: dict, quantizer_name: str) -> dict:
    """Copy one quantizer's numeric config from a native ModelOpt preset."""
    entries = mtq.normalize_quant_cfg_list(copy.deepcopy(preset["quant_cfg"]))
    return copy.deepcopy(next(entry["cfg"] for entry in entries if entry["quantizer_name"] == quantizer_name))


def _mxfp8_quantize_config() -> tuple[list[dict], None]:
    """ModelOpt MXFP8 numerics executed by verl's export-shared TileLang kernel."""
    quant_cfg = mtq.normalize_quant_cfg_list(copy.deepcopy(mtq.MXFP8_DEFAULT_CFG["quant_cfg"]))
    for entry in quant_cfg:
        cfg = entry.get("cfg")
        if cfg is not None and entry.get("enable", True):
            cfg["backend"] = _MXFP8_BACKEND
    return quant_cfg, None


def _mxfp4_quantize_config(*, dense_mxfp8: bool) -> tuple[list[dict], None]:
    """MXFP4 routed-expert weights with MXFP8 activations.

    ModelOpt has native MXFP4/MXFP4 and MXFP8/MXFP8 presets but no MXFP4/MXFP8
    preset. Their quantizers are independent, so combine the weight numerics
    from the former with the input numerics from the latter.
    """
    if dense_mxfp8:
        quant_cfg, _ = _mxfp8_quantize_config()
    else:
        quant_cfg = [{"quantizer_name": "*", "enable": False}]

    weight_cfg = _modelopt_quantizer_cfg(mtq.MXFP4_DEFAULT_CFG, "*weight_quantizer")
    weight_cfg["backend"] = _MXFP4_BACKEND
    input_cfg = _modelopt_quantizer_cfg(mtq.MXFP8_DEFAULT_CFG, "*input_quantizer")
    input_cfg["backend"] = _MXFP8_BACKEND
    quant_cfg.extend(
        [
            {
                "quantizer_name": _ROUTED_EXPERT_WEIGHT_QUANTIZER,
                "cfg": weight_cfg,
            },
            {
                "quantizer_name": _ROUTED_EXPERT_INPUT_QUANTIZER,
                "cfg": input_cfg,
            },
        ]
    )
    return quant_cfg, None


def _nvfp4_quantize_config() -> tuple[list[dict], None]:
    """NVFP4 W4A4 on routed experts, with the rollout's per-token activation scales."""
    quant_cfg = mtq.normalize_quant_cfg_list(copy.deepcopy(mtq.NVFP4_EXPERTS_ONLY_CFG["quant_cfg"]))
    for entry in quant_cfg:
        cfg = entry.get("cfg")
        if cfg is None or not entry.get("enable", True):
            continue
        name = entry["quantizer_name"]
        if name.endswith("weight_quantizer"):
            cfg["backend"] = _NVFP4_WEIGHT_BACKEND
        elif name.endswith("input_quantizer"):
            cfg["backend"] = _NVFP4_INPUT_BACKEND
    return quant_cfg, None


def _ignore_patterns_to_quant_cfg(ignore_patterns: list[str]) -> list[dict]:
    cfg = []
    mapping = {
        "lm_head": "*output_layer*",
        "*mlp.gate": "*router*",
        "*self_attn*": "*self_attention*",
    }
    for pattern in ignore_patterns:
        key = mapping.get(pattern, pattern)
        cfg.append({"quantizer_name": key, "enable": False})
        # Module patterns must also match their child quantizers. Keep the
        # original rule for callers that already name quantizers explicitly.
        cfg.append({"quantizer_name": f"{key}.*", "enable": False})
    return cfg


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    """Build a complete ModelOpt quantization config for ``mtq.quantize``."""
    if qat_mode not in _SUPPORTED_QAT_MODES:
        raise ValueError(f"Supported QAT modes are {sorted(_SUPPORTED_QAT_MODES)}, got: {qat_mode}")

    if ignore_patterns is None:
        ignore_patterns = []

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    if qat_mode == "mxfp8":
        quant_cfg, algorithm = _mxfp8_quantize_config()
    elif qat_mode in ("mxfp4", "mxfp4_mxfp8"):
        quant_cfg, algorithm = _mxfp4_quantize_config(dense_mxfp8=qat_mode == "mxfp4_mxfp8")
    elif qat_mode == "nvfp4":
        quant_cfg, algorithm = _nvfp4_quantize_config()
    else:
        quant_cfg = mtq.normalize_quant_cfg_list(copy.deepcopy(_QUANTIZER_CFGS[qat_mode]))
        disabled_cfg = copy.deepcopy(_default_disabled_quantizer_cfg)
        if isinstance(disabled_cfg, dict):
            disabled_cfg = mtq.normalize_quant_cfg_list(disabled_cfg)
        quant_cfg.extend(disabled_cfg)
        algorithm = None
    quant_cfg.extend(ignore_cfg)
    return {"quant_cfg": quant_cfg, "algorithm": algorithm}


# ---------------------------------------------------------------------------
# Patch ModelOpt's grouped-linear weight fake quant
# ---------------------------------------------------------------------------
class _PrequantizedWeightsView:
    """A grouped-linear view that prevents ModelOpt from quantizing weights twice."""

    def __init__(self, module: nn.Module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    @staticmethod
    def weight_quantizer(weight: torch.Tensor) -> torch.Tensor:
        return weight


def _grouped_weight_quantizers_are_compatible(quantizers: GroupedQuantizer) -> bool:
    """Whether per-expert quantizers can share one stateless batched invocation."""
    if not quantizers or not all(isinstance(quantizer, TensorQuantizer) for quantizer in quantizers):
        return False
    first = quantizers[0]
    if first.backend not in _GROUPABLE_WEIGHT_BACKENDS:
        return False
    return all(
        quantizer.is_enabled
        and quantizer.fake_quant
        and not getattr(quantizer, "_if_calib", False)
        and quantizer.backend == first.backend
        and quantizer.block_sizes == first.block_sizes
        and quantizer.backend_extra_args == first.backend_extra_args
        for quantizer in quantizers
    )


def _assert_grouped_scale_boundaries(weights: tuple[torch.Tensor, ...], quantizer: TensorQuantizer) -> None:
    """Assert that every expert consists of complete, independent scale blocks."""
    assert weights, "grouped fake quant requires at least one expert weight"
    shape = weights[0].shape
    assert len(shape) == 2, f"grouped fake quant expects 2D expert weights, got {shape}"
    assert all(weight.shape == shape for weight in weights), "grouped fake quant requires equal expert shapes"

    if quantizer.backend == _FP8_BLOCK_BACKEND:
        block_rows, block_cols = quantizer.backend_extra_args["block_sizes"]
    else:
        assert quantizer.backend in (_MXFP8_BACKEND, _MXFP4_BACKEND)
        assert quantizer.block_sizes.get("type") == "dynamic"
        block_rows, block_cols = 1, quantizer.block_sizes[-1]

    rows, cols = shape
    assert rows % block_rows == 0, (
        f"expert rows ({rows}) must be divisible by scale block rows ({block_rows}); "
        "a scale block must not cross an expert boundary"
    )
    assert cols % block_cols == 0, f"expert columns ({cols}) must be divisible by scale block columns ({block_cols})"


def _group_fake_quant_weights(
    weights: tuple[torch.Tensor, ...],
    quantizers: GroupedQuantizer,
) -> tuple[torch.Tensor, ...]:
    """Fake-quantize separate expert weights in one pointer-table kernel launch."""
    first = quantizers[0]
    _assert_grouped_scale_boundaries(weights, first)
    if first.backend == _FP8_BLOCK_BACKEND:
        block_rows, block_cols = first.backend_extra_args["block_sizes"]
        quantized = _GroupedFp8FakeQuant.apply(block_rows, block_cols, None, *weights)
    elif first.backend == _MXFP8_BACKEND:
        quantized = _GroupedFp8FakeQuant.apply(1, first.block_sizes[-1], "e8m0", *weights)
    else:
        assert first.backend == _MXFP4_BACKEND
        quantized = _GroupedMxfp4FakeQuant.apply(*weights)
    assert quantized.shape == (len(weights), *weights[0].shape)
    return quantized.unbind(0)


def _patch_te_grouped_weight_fake_quant() -> None:
    """Batch verl's stateless expert-weight QDQ backends inside TE grouped linears."""
    try:
        from modelopt.torch.quantization.plugins.transformer_engine import _QuantTEGroupedLinear
    except ImportError:
        return

    original = _QuantTEGroupedLinear.__dict__.get("_quantized_linear_fn")
    original_fn = getattr(original, "__func__", original)
    if original_fn is None or getattr(original_fn, "_verl_grouped_weight_fake_quant", False):
        return

    def grouped_quantized_linear_fn(package, func_name, module, *args):
        quantizers = module.weight_quantizer
        num_gemms = module.num_gemms
        if (
            not isinstance(quantizers, GroupedQuantizer)
            or len(quantizers) != num_gemms
            or not _grouped_weight_quantizers_are_compatible(quantizers)
        ):
            return original_fn(package, func_name, module, *args)

        weights_start = len(args) - 2 * num_gemms
        weights = tuple(args[weights_start : weights_start + num_gemms])
        if len(weights) != num_gemms or len({(weight.shape, weight.dtype, weight.device) for weight in weights}) != 1:
            return original_fn(package, func_name, module, *args)

        quantized_weights = _group_fake_quant_weights(weights, quantizers)
        new_args = (*args[:weights_start], *quantized_weights, *args[weights_start + num_gemms :])
        return original_fn(package, func_name, _PrequantizedWeightsView(module), *new_args)

    grouped_quantized_linear_fn._verl_grouped_weight_fake_quant = True
    _QuantTEGroupedLinear._quantized_linear_fn = staticmethod(grouped_quantized_linear_fn)


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    config = build_quantize_config(qat_mode, ignore_patterns)
    _patch_te_grouped_weight_fake_quant()
    mtq.quantize(model, config)
    return model


def _get_qat_field(qat_config, key, default=None):
    """Extract a field from qat_config, supporting both dict and object-style access."""
    if isinstance(qat_config, dict):
        return qat_config.get(key, default)
    return getattr(qat_config, key, default)


def apply_qat_to_modules(modules, qat_config):
    """Apply ModelOpt fake quantization to a list of Megatron module chunks."""
    qat_mode = _get_qat_field(qat_config, "mode", "fp8")
    if qat_mode == "nvfp4":
        from megatron.core import parallel_state

        # The export computes each expert's global scale over the whole matrix.
        if parallel_state.get_expert_tensor_parallel_world_size() > 1:
            raise NotImplementedError("NVFP4 QAT needs expert_tensor_parallel_size=1")
    ignore_patterns = _get_qat_field(qat_config, "ignore_patterns", None)
    if ignore_patterns is not None:
        ignore_patterns = list(ignore_patterns)

    for i in range(len(modules)):
        modules[i] = apply_qat(modules[i], qat_mode, ignore_patterns=ignore_patterns)
    return modules
