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
import logging

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    is_registered_quant_backend,
    register_quant_backend,
)

logger = logging.getLogger(__name__)

_FP8_BLOCK_BACKEND = "verl_fp8_block"
_MXFP4_BACKEND = "verl_mxfp4"
_NVFP4_WEIGHT_BACKEND = "verl_nvfp4_weight"
_NVFP4_INPUT_BACKEND = "verl_nvfp4_input"
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


class _FP8BlockFakeQuant(torch.autograd.Function):
    """Dynamic FP8 E4M3 fake quantization over blocks of the last two dims, with straight-through grad."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, block_rows: int, block_cols: int) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).float()
        rows, cols = x2d.shape
        pad_rows, pad_cols = -rows % block_rows, -cols % block_cols
        xp = F.pad(x2d, (0, pad_cols, 0, pad_rows))
        blocks = xp.reshape(xp.shape[0] // block_rows, block_rows, xp.shape[1] // block_cols, block_cols)
        scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / _FP8_MAX
        dq = (blocks / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float() * scale
        dq = dq.reshape(xp.shape)[:rows, :cols]
        return dq.reshape(shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


def _fp8_block_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    block_rows, block_cols = quantizer.backend_extra_args["block_sizes"]
    return _FP8BlockFakeQuant.apply(inputs, block_rows, block_cols)


class _Mxfp4FakeQuant(torch.autograd.Function):
    """compressed-tensors-compatible MXFP4 fake quantization with STE."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        from compressed_tensors.quantization import QuantizationArgs
        from compressed_tensors.quantization.lifecycle.forward import fake_quantize
        from compressed_tensors.quantization.utils import calculate_qparams

        args = QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            group_size=32,
            strategy="group",
            dynamic=False,
            scale_dtype="torch.uint8",
        )
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        groups = x2d.unflatten(-1, (-1, 32))
        min_vals, max_vals = groups.amin(dim=-1), groups.amax(dim=-1)
        scale, zero_point = calculate_qparams(min_vals.float(), max_vals.float(), args)
        return fake_quantize(x2d, scale, zero_point, args).reshape(shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad


def _mxfp4_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    return _Mxfp4FakeQuant.apply(inputs)


class _Nvfp4FakeQuant(torch.autograd.Function):
    """compressed-tensors-compatible NVFP4 QDQ with an explicit global scale."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
        from compressed_tensors.quantization import QuantizationArgs
        from compressed_tensors.quantization.lifecycle.forward import fake_quantize
        from compressed_tensors.quantization.utils import calculate_qparams

        args = QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            group_size=16,
            strategy="tensor_group",
            dynamic=False,
            scale_dtype="torch.float8_e4m3fn",
        )
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        groups = x2d.unflatten(-1, (-1, 16))
        min_vals, max_vals = groups.amin(dim=-1), groups.amax(dim=-1)
        scale, zero_point = calculate_qparams(
            min_vals.float(),
            max_vals.float(),
            args,
            global_scale=global_scale.float(),
        )
        return (
            fake_quantize(
                x2d,
                scale,
                zero_point,
                args,
                global_scale=global_scale.float(),
            )
            .reshape(shape)
            .to(x.dtype)
        )

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _nvfp4_global_scale(inputs: torch.Tensor) -> torch.Tensor:
    from compressed_tensors.quantization.utils import generate_gparam

    amax = inputs.detach().float().abs().amax()
    return generate_gparam(-amax, amax).to(device=inputs.device)


def _nvfp4_weight_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    if inputs.ndim > 2:
        # TEGroupedLinear stores one expert matrix per leading entry. The
        # compressed-tensors checkpoint likewise assigns one global scale per
        # expert, so quantize each matrix independently.
        return torch.stack(
            [_Nvfp4FakeQuant.apply(weight, _nvfp4_global_scale(weight)) for weight in inputs],
            dim=0,
        )
    return _Nvfp4FakeQuant.apply(inputs, _nvfp4_global_scale(inputs))


def _nvfp4_input_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    # vLLM's W4A4 kernel needs one static tensor-level scale while the local
    # group-16 scales remain dynamic. Keep a running maximum during QAT and use
    # that exact scale for both fake quant and checkpoint export.
    current = _nvfp4_global_scale(inputs)
    previous = getattr(quantizer, "_verl_input_global_scale", None)
    if previous is not None:
        # Running max of amax == running min of the global divisor.
        current = torch.minimum(previous.to(current.device), current)
    quantizer._verl_input_global_scale = current.detach()
    return _Nvfp4FakeQuant.apply(inputs, current)


if not is_registered_quant_backend(_FP8_BLOCK_BACKEND):
    register_quant_backend(_FP8_BLOCK_BACKEND, _fp8_block_backend)
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


def _modelopt_quantizer_cfg(preset: dict, quantizer_name: str) -> dict:
    """Copy one quantizer's numeric config from a native ModelOpt preset."""
    entries = mtq.normalize_quant_cfg_list(copy.deepcopy(preset["quant_cfg"]))
    return copy.deepcopy(next(entry["cfg"] for entry in entries if entry["quantizer_name"] == quantizer_name))


def _mxfp4_quantize_config(*, dense_mxfp8: bool) -> tuple[list[dict], None]:
    """MXFP4 routed-expert weights with MXFP8 activations.

    ModelOpt has native MXFP4/MXFP4 and MXFP8/MXFP8 presets but no MXFP4/MXFP8
    preset. Their quantizers are independent, so combine the weight numerics
    from the former with the input numerics from the latter.
    """
    if dense_mxfp8:
        quant_cfg = mtq.normalize_quant_cfg_list(copy.deepcopy(mtq.MXFP8_DEFAULT_CFG["quant_cfg"]))
    else:
        quant_cfg = [{"quantizer_name": "*", "enable": False}]

    weight_cfg = _modelopt_quantizer_cfg(mtq.MXFP4_DEFAULT_CFG, "*weight_quantizer")
    # ModelOpt and compressed-tensors use different E2M1 tie handling for a
    # small subset of values. Export is compressed-tensors, so QAT must use the
    # same QDQ implementation to remain numerically identical.
    weight_cfg["backend"] = _MXFP4_BACKEND
    quant_cfg.extend(
        [
            {
                "quantizer_name": _ROUTED_EXPERT_WEIGHT_QUANTIZER,
                "cfg": weight_cfg,
            },
            {
                "quantizer_name": _ROUTED_EXPERT_INPUT_QUANTIZER,
                "cfg": _modelopt_quantizer_cfg(mtq.MXFP8_DEFAULT_CFG, "*input_quantizer"),
            },
        ]
    )
    return quant_cfg, None


def _nvfp4_quantize_config() -> tuple[list[dict], None]:
    """NVFP4 W4A4 on routed experts with export-compatible running scales."""
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


def collect_nvfp4_input_global_scales(model: nn.Module | list[nn.Module]) -> dict[str, torch.Tensor]:
    """Collect observed expert activation scales under HF-style canonical keys."""
    scales = {}
    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        for name, module in chunk.named_modules():
            scale = getattr(module, "_verl_input_global_scale", None)
            if scale is None or not name.endswith("input_quantizer") or ".mlp.experts" not in name:
                continue
            layer = next((part for part in name.split(".") if part.isdigit()), None)
            projection = "gate_up" if "linear_fc1" in name else "down" if "linear_fc2" in name else None
            if layer is not None and projection is not None:
                scales[f"layers.{layer}.mlp.experts.{projection}"] = scale.detach()
    return scales


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
        # ModelOpt's native preset is exactly the compressed-tensors MXFP8
        # numerics: dynamic groups of 32, E4M3 values and E8M0 scales.
        modelopt_config = copy.deepcopy(mtq.MXFP8_DEFAULT_CFG)
        quant_cfg = mtq.normalize_quant_cfg_list(modelopt_config["quant_cfg"])
        algorithm = modelopt_config["algorithm"]
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


class _QuantizedWeightsView:
    """The grouped linear module as ModelOpt's functional sees it, with the expert weights already quantized."""

    def __init__(self, module: nn.Module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    @staticmethod
    def weight_quantizer(weight: torch.Tensor) -> torch.Tensor:
        return weight


def _batch_grouped_expert_weight_qdq() -> None:
    """Fake-quantize all expert weights of a TE grouped linear in one quantizer call.

    ModelOpt's ``_QuantTEGroupedLinear`` calls its weight quantizer once per expert at every forward. With MXFP4
    that is about fifty small kernels per expert, launched from Python: on Qwen3-30B-A3B at TP2 x EP2 (64 local
    experts), a QAT forward of one layer took 147 ms against 7 ms without QAT, with the GPU idle most of the time.
    The MXFP4 QDQ forms each 32-element group within a row of one weight, so quantizing the stacked weights gives
    the same values, bit for bit, as quantizing them one by one. Other weight backends keep ModelOpt's per-expert
    loop.
    """
    try:
        from modelopt.torch.quantization.plugins.transformer_engine import _QuantTEGroupedLinear
    except ImportError:
        return

    original = _QuantTEGroupedLinear.__dict__.get("_quantized_linear_fn")
    original_fn = getattr(original, "__func__", original)
    if original_fn is None:
        logger.warning(
            "ModelOpt's _QuantTEGroupedLinear has no _quantized_linear_fn; MXFP4 expert weights stay per expert"
        )
        return
    if getattr(original_fn, "_verl_batched_weight_qdq", False):
        return

    def quantized_linear_fn(package, func_name, self, *args):
        quantizer = self.weight_quantizer
        num_gemms = getattr(self, "num_gemms", 0)
        batchable = getattr(quantizer, "is_enabled", False) and getattr(quantizer, "backend", None) == _MXFP4_BACKEND
        if not batchable or num_gemms < 2 or 2 * num_gemms > len(args):
            return original_fn(package, func_name, self, *args)
        # *weights_and_biases is the trailing variadic of TE's _GroupedLinear: num_gemms weights, then num_gemms biases
        start = len(args) - 2 * num_gemms
        weights = args[start : start + num_gemms]
        if len({(w.shape, w.dtype, w.device) for w in weights}) != 1:
            return original_fn(package, func_name, self, *args)
        quantized = quantizer(torch.stack(weights)).unbind(0)
        args = (*args[:start], *quantized, *args[start + num_gemms :])
        return original_fn(package, func_name, _QuantizedWeightsView(self), *args)

    quantized_linear_fn._verl_batched_weight_qdq = True
    _QuantTEGroupedLinear._quantized_linear_fn = staticmethod(quantized_linear_fn)
    logger.info("QAT: TE grouped linears fake-quantize their MXFP4 expert weights in one call")


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    config = build_quantize_config(qat_mode, ignore_patterns)
    _batch_grouped_expert_weight_qdq()
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
    ignore_patterns = _get_qat_field(qat_config, "ignore_patterns", None)
    if ignore_patterns is not None:
        ignore_patterns = list(ignore_patterns)

    for i in range(len(modules)):
        modules[i] = apply_qat(modules[i], qat_mode, ignore_patterns=ignore_patterns)
    return modules
