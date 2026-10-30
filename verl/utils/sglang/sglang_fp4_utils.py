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
"""SGLang MXFP4/NVFP4 MoE staging and in-place post-load refit utilities."""

from __future__ import annotations

import torch

MXFP4_BLOCK_SIZE = 32

# ---------------------------------------------------------------------------
# MXFP4 experts
# ---------------------------------------------------------------------------

_MXFP4_PARAMS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_bias",
    "w2_weight_bias",
)
_MXFP4_DERIVED_PARAMS = ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")
_NVFP4_PARAMS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
_NVFP4_DERIVED_PARAMS = (
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "g1_alphas",
    "g2_alphas",
    "w13_input_scale_quant",
    "w2_input_scale_quant",
)


def _is_mxfp4_moe(module: torch.nn.Module) -> bool:
    qm = getattr(module, "quant_method", None)
    return qm is not None and (type(qm).__name__ == "Mxfp4MoEMethod" or getattr(qm, "_verl_mxfp4_moe_method", False))


def _mxfp4_checkpoint_layout(module: torch.nn.Module) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Canonical compressed-tensors layout consumed by ``FusedMoE.weight_loader``."""
    qm = module.quant_method
    experts = module.num_local_experts
    intermediate = qm.intermediate_size_per_partition
    hidden = qm.hidden_size
    return {
        "w13_weight": ((experts, 2 * intermediate, hidden // 2), torch.uint8),
        "w2_weight": ((experts, hidden, intermediate // 2), torch.uint8),
        "w13_weight_scale": ((experts, 2 * intermediate, hidden // MXFP4_BLOCK_SIZE), torch.uint8),
        "w2_weight_scale": ((experts, hidden, intermediate // MXFP4_BLOCK_SIZE), torch.uint8),
        "w13_weight_bias": ((experts, 2 * intermediate), torch.bfloat16),
        "w2_weight_bias": ((experts, hidden), torch.bfloat16),
    }


def _mxfp4_staging_data(data: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, *, zero: bool) -> torch.Tensor:
    """Return a checkpoint-layout view when it spans the live storage, otherwise allocate."""
    shape = torch.Size(shape)
    if data.dtype == dtype and data.shape == shape:
        return data
    if data.is_contiguous() and data.numel() * data.element_size() == shape.numel() * dtype.itemsize:
        return data.flatten().view(dtype).reshape(shape)
    factory = torch.zeros if zero else torch.empty
    return factory(shape, dtype=dtype, device=data.device)


def stage_mxfp4_moe_params(model: torch.nn.Module):
    """Expose canonical MXFP4 expert tensors for all buckets of one refit.

    The packed weights and scales generally reuse the live kernel storage by a
    dtype view. Biases are zeroed canonical buffers because SGLang creates them
    for the SM100 FlashInfer ABI even when the HF model has no expert biases.
    """
    pending = []
    for name, module in model.named_modules():
        if not _is_mxfp4_moe(module):
            continue
        qm = module.quant_method
        supported = (
            getattr(qm, "use_flashinfer", False) and getattr(qm, "_fi_kernel", None) == "trtllm_sm100"
        ) or getattr(qm, "use_deep_gemm", False)
        if not supported:
            raise NotImplementedError(
                f"{name}: SGLang MXFP4 refit supports the Blackwell flashinfer_mxfp4 and deep_gemm backends; "
                f"got _fi_kernel={getattr(qm, '_fi_kernel', None)!r}, "
                f"use_deep_gemm={getattr(qm, 'use_deep_gemm', False)}"
            )
        params = {}
        for pname, (shape, dtype) in _mxfp4_checkpoint_layout(module).items():
            param = getattr(module, pname, None)
            if not isinstance(param, torch.nn.Parameter):
                continue
            staging = _mxfp4_staging_data(param.data, shape, dtype, zero=pname.endswith("_bias"))
            params[pname] = (param, param.data, staging)
        derived = {
            pname: getattr(module, pname)
            for pname in _MXFP4_DERIVED_PARAMS
            if isinstance(getattr(module, pname, None), torch.Tensor)
        }
        pending.append((name, module, params, derived))

    # Allocate everything before rebinding anything, so OOM cannot leave a
    # partially staged model whose state was never returned to the caller.
    for _, module, params, _ in pending:
        for pname, (param, _, staging) in params.items():
            param.data = staging
            setattr(module, pname, param)
            if pname in ("w13_weight", "w2_weight"):
                # compressed-tensors exports ``weight_packed``. Qwen3.5 keeps
                # that suffix while mapping per-expert names to the fused
                # parameter, so publish the checkpoint alias for params_dict.
                setattr(module, f"{pname}_packed", param)
    return pending


def restore_mxfp4_live(staged) -> None:
    for _, module, params, derived in staged:
        for pname, (param, live, _) in params.items():
            if pname in ("w13_weight", "w2_weight") and hasattr(module, f"{pname}_packed"):
                delattr(module, f"{pname}_packed")
            param.data = live
            setattr(module, pname, param)
        for pname, live in derived.items():
            setattr(module, pname, live)


def reprocess_mxfp4_moe_layers(staged) -> int:
    """Rebuild MXFP4 kernel layouts and fold them into CUDA-graph-captured storage."""
    count = 0
    for name, module, params, derived in staged:
        module.quant_method.process_weights_after_loading(module)
        for pname, (param, live, _) in params.items():
            rebuilt = getattr(module, pname, None)
            if not isinstance(rebuilt, torch.Tensor):
                raise RuntimeError(f"{name}.{pname}: MXFP4 post-load processing removed the expert tensor")
            if rebuilt.shape != live.shape or rebuilt.dtype != live.dtype:
                raise RuntimeError(
                    f"{name}.{pname}: MXFP4 post-load processing produced "
                    f"{tuple(rebuilt.shape)}/{rebuilt.dtype}, but CUDA graphs captured "
                    f"{tuple(live.shape)}/{live.dtype}"
                )
            if rebuilt.data_ptr() != live.data_ptr():
                live.copy_(rebuilt)
            if pname in ("w13_weight", "w2_weight") and hasattr(module, f"{pname}_packed"):
                delattr(module, f"{pname}_packed")
            param.data = live
            setattr(module, pname, param)
        for pname, live in derived.items():
            rebuilt = getattr(module, pname, None)
            if not isinstance(rebuilt, torch.Tensor) or rebuilt.shape != live.shape or rebuilt.dtype != live.dtype:
                raise RuntimeError(f"{name}.{pname}: MXFP4 post-load processing changed a captured derived tensor")
            if rebuilt.data_ptr() != live.data_ptr():
                live.data.copy_(rebuilt.data)
            setattr(module, pname, live)
        count += 1
    return count


# ---------------------------------------------------------------------------
# NVFP4 experts
# ---------------------------------------------------------------------------


def _is_nvfp4_moe(module: torch.nn.Module) -> bool:
    scheme = getattr(module, "scheme", None)
    return scheme is not None and (
        type(scheme).__name__ == "CompressedTensorsW4A4Nvfp4MoE" or getattr(scheme, "_verl_nvfp4_moe_scheme", False)
    )


def stage_nvfp4_moe_params(model: torch.nn.Module):
    """Expose compressed-tensors NVFP4 checkpoint tensors during one refit."""
    pending = []
    for name, module in model.named_modules():
        if not _is_nvfp4_moe(module):
            continue
        scheme = module.scheme
        if getattr(scheme, "use_flashinfer_trtllm", False):
            raise NotImplementedError(f"{name}: SGLang NVFP4 refit supports only the flashinfer_cutlass backend")

        params = {}
        for pname in _NVFP4_PARAMS:
            param = getattr(module, pname, None)
            if not isinstance(param, torch.nn.Parameter):
                raise RuntimeError(f"{name}.{pname}: SGLang NVFP4 expert parameter is missing")
            weight_loader = getattr(module, "weight_loader", None)
            if weight_loader is None:
                weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                raise RuntimeError(f"{name}.{pname}: SGLang NVFP4 expert weight loader is missing")
            param.weight_loader = weight_loader
            if pname.endswith("_weight_scale"):
                param.quant_method = "group"
            # CUTLASS keeps weights byte-compatible and only swizzles block
            # scales. Reusing the live allocation provides the canonical
            # checkpoint buffer without doubling expert memory.
            staging = _mxfp4_staging_data(param.data, tuple(param.shape), param.dtype, zero=False)
            params[pname] = (param, param.data, staging)
        derived = {
            pname: getattr(module, pname)
            for pname in _NVFP4_DERIVED_PARAMS
            if isinstance(getattr(module, pname, None), torch.Tensor)
        }
        pending.append((name, module, params, derived))

    for _, module, params, _ in pending:
        for pname, (param, _, staging) in params.items():
            param.data = staging
            setattr(module, pname, param)
            if pname in ("w13_weight", "w2_weight"):
                setattr(module, f"{pname}_packed", param)
    return pending


def restore_nvfp4_live(staged) -> None:
    for _, module, params, derived in staged:
        for pname, (param, live, _) in params.items():
            if pname in ("w13_weight", "w2_weight") and hasattr(module, f"{pname}_packed"):
                delattr(module, f"{pname}_packed")
            param.data = live
            setattr(module, pname, param)
        for pname, live in derived.items():
            setattr(module, pname, live)


def reprocess_nvfp4_moe_layers(staged) -> int:
    """Rebuild NVFP4 CUTLASS scales inside CUDA-graph-captured storage."""
    count = 0
    for name, module, params, derived in staged:
        module.quant_method.process_weights_after_loading(module)
        for pname, (param, live, _) in params.items():
            rebuilt = getattr(module, pname, None)
            if not isinstance(rebuilt, torch.Tensor) or rebuilt.shape != live.shape or rebuilt.dtype != live.dtype:
                raise RuntimeError(f"{name}.{pname}: NVFP4 post-load processing changed a captured expert tensor")
            if rebuilt.data_ptr() != live.data_ptr():
                live.copy_(rebuilt)
            if pname in ("w13_weight", "w2_weight") and hasattr(module, f"{pname}_packed"):
                delattr(module, f"{pname}_packed")
            param.data = live
            setattr(module, pname, param)
        for pname, live in derived.items():
            rebuilt = getattr(module, pname, None)
            if not isinstance(rebuilt, torch.Tensor) or rebuilt.shape != live.shape or rebuilt.dtype != live.dtype:
                raise RuntimeError(f"{name}.{pname}: NVFP4 post-load processing changed a captured derived tensor")
            if rebuilt.data_ptr() != live.data_ptr():
                live.data.copy_(rebuilt.data)
            setattr(module, pname, live)
        count += 1
    return count
