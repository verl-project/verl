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
"""SGLang MXFP8 staging and in-place post-load refit utilities.

SGLang derives backend-specific scale layouts during model loading. Bucketed
weight sync restores canonical scales, loads every bucket, then calls these
helpers once to rebuild those layouts and copy them back into the storage
captured by CUDA Graphs.
"""

from __future__ import annotations

import torch

MXFP8_BLOCK_SIZE = 32


def _is_mxfp8_linear(module: torch.nn.Module) -> bool:
    qm = getattr(module, "quant_method", None)
    return (
        qm is not None
        and getattr(qm, "use_mxfp8", False)
        and hasattr(qm, "process_weights_after_loading")
        and hasattr(module, "weight_scale_inv")
    )


def _layer_backend(qm) -> tuple[object, bool]:
    """Return ``(backend, resolved)`` for one MXFP8 quant method.

    ``resolved`` is True when sglang itself resolved the MXFP8 dense backend
    (>= 0.5.18, ``quant_method.mxfp8_dense_backend``); otherwise the requested
    ``--fp8-gemm-backend`` is returned, where ``auto`` means Triton for MXFP8.
    """
    backend = getattr(qm, "mxfp8_dense_backend", None)
    if backend is not None:
        return backend, True
    from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend

    return get_fp8_gemm_runner_backend(), False


def _needs_reprocess(backend, resolved: bool) -> bool:
    if backend.is_flashinfer_trtllm():
        raise NotImplementedError(
            "MXFP8 weight sync with the flashinfer_trtllm GEMM backend is not supported: that backend "
            "shuffles the weight tensor in place at load time, so post-load processing cannot be re-run "
            "after a sync. Use flashinfer_cutlass / flashinfer_cutedsl, or leave the backend unset."
        )
    if resolved:
        # CUTLASS, CuTe-DSL and DeepGEMM all derive their kernel layout from the canonical
        # weight_scale_inv the sync just rewrote; re-running the processing is idempotent for
        # them and a no-op for backends that read the canonical scale directly.
        return True
    # sglang <= 0.5.17: only flashinfer_cutlass keeps a derived copy; Triton (auto) reads
    # weight_scale_inv directly.
    return backend.is_flashinfer_cutlass()


def reprocess_mxfp8_layers(model: torch.nn.Module) -> int:
    """Re-derive backend-specific MXFP8 scale layouts from the canonical scales.

    Returns the number of modules reprocessed. Raises for backends whose
    post-load processing is not idempotent (``flashinfer_trtllm``).
    """
    count = 0
    for name, module in model.named_modules():
        if not _is_mxfp8_linear(module):
            continue
        qm = module.quant_method
        backend, resolved = _layer_backend(qm)
        if not _needs_reprocess(backend, resolved):
            continue
        if not getattr(qm, "is_checkpoint_fp8_serialized", True):
            # Would re-quantize from a bf16 master weight that a sync just replaced with fp8 data.
            raise RuntimeError(
                f"{name}: MXFP8 quant method expects a non-fp8-serialized checkpoint; refusing to "
                "re-quantize after a weight sync."
            )
        # DeepGEMM can fall back to FlashInfer for unsupported shapes. SGLang clears
        # the fallback's swizzled scale before rebuilding it, which would otherwise
        # rebind storage that CUDA graphs captured at startup.
        captured = {
            attr: value
            for attr in ("weight_scale_inv_swizzled", "weight_scale_inv_deepgemm")
            if isinstance((value := getattr(module, attr, None)), torch.Tensor)
        }
        qm.process_weights_after_loading(module)
        for attr, live in captured.items():
            rebuilt = getattr(module, attr, None)
            if not isinstance(rebuilt, torch.Tensor):
                raise RuntimeError(f"{name}.{attr}: post-load processing did not rebuild the captured scale")
            if rebuilt.shape != live.shape or rebuilt.dtype != live.dtype:
                raise RuntimeError(
                    f"{name}.{attr}: post-load processing produced {tuple(rebuilt.shape)}/{rebuilt.dtype} but "
                    f"CUDA graphs captured {tuple(live.shape)}/{live.dtype}"
                )
            if rebuilt.data_ptr() != live.data_ptr():
                live.copy_(rebuilt)
                setattr(module, attr, live)
        count += 1
    return count


# ---------------------------------------------------------------------------
# MoE experts
# ---------------------------------------------------------------------------

_MOE_SCALE_PAIRS = (("w13_weight", "w13_weight_scale_inv"), ("w2_weight", "w2_weight_scale_inv"))


def _is_mxfp8_moe(module: torch.nn.Module) -> bool:
    qm = getattr(module, "quant_method", None)
    return (
        qm is not None
        and getattr(qm, "use_mxfp8", False)
        and hasattr(qm, "process_weights_after_loading")
        and hasattr(module, "w13_weight_scale_inv")
    )


def _canonical_scale_shape(weight: torch.Tensor) -> tuple[int, ...]:
    return (*tuple(weight.shape[:-1]), weight.shape[-1] // MXFP8_BLOCK_SIZE)


def stage_mxfp8_moe_scales(model: torch.nn.Module) -> list[tuple[torch.nn.Module, str, torch.Tensor]]:
    """Expose canonical-layout expert scale buffers so ``load_weights`` can write into them.

    Returns ``(module, scale_name, live_data)`` for every expert scale whose live
    buffer no longer has the checkpoint shape or dtype; ``reprocess_mxfp8_moe_layers``
    folds the re-derived layout back into ``live_data``. Scales the runner rewrote
    without changing shape or dtype (FlashInfer TRT-LLM; CUTLASS on sglang <= 0.5.15)
    or left canonical (CUTLASS on sglang >= 0.5.20) are not staged: ``load_weights``
    writes the canonical data over the live buffer and the re-processing rewrites it
    in place, if at all.
    """
    pending = []
    for module in model.modules():
        if not _is_mxfp8_moe(module):
            continue
        for wname, sname in _MOE_SCALE_PAIRS:
            weight = getattr(module, wname, None)
            param = getattr(module, sname, None)
            if weight is None or not isinstance(param, torch.nn.Parameter):
                continue
            canonical = _canonical_scale_shape(weight.data)
            if tuple(param.data.shape) == canonical and param.data.dtype == torch.uint8:
                continue
            staging = torch.empty(canonical, dtype=torch.uint8, device=param.data.device)
            pending.append((module, sname, param, staging))
    # Rebind only once every staging buffer is allocated: a failed allocation must not leave a
    # parameter on a staging buffer that no returned list records.
    staged = []
    for module, sname, param, staging in pending:
        staged.append((module, sname, param.data))
        param.data = staging
    return staged


def reprocess_mxfp8_moe_layers(model: torch.nn.Module, staged: list[tuple[torch.nn.Module, str, torch.Tensor]]) -> int:
    """Re-run the MoE post-load processing and fold results into the captured storage."""
    live_by_key = {(id(m), n): live for m, n, live in staged}
    count = 0
    for name, module in model.named_modules():
        if not _is_mxfp8_moe(module):
            continue
        qm = module.quant_method
        if not getattr(qm, "is_checkpoint_fp8_serialized", True):
            raise RuntimeError(
                f"{name}: MXFP8 MoE quant method expects a non-fp8-serialized checkpoint; refusing to "
                "re-quantize experts after a weight sync."
            )
        qm.process_weights_after_loading(module)
        for _, sname in _MOE_SCALE_PAIRS:
            live = live_by_key.get((id(module), sname))
            if live is None:
                continue
            param = getattr(module, sname)
            new = param.data
            if tuple(new.shape) != tuple(live.shape) or new.dtype != live.dtype:
                raise RuntimeError(
                    f"{name}.{sname}: post-load processing produced {tuple(new.shape)}/{new.dtype} but the live "
                    f"buffer is {tuple(live.shape)}/{live.dtype}; the MoE runner's scale layout changed between "
                    "load and refit and cannot be updated in place."
                )
            if new.data_ptr() != live.data_ptr():
                live.copy_(new)
            param.data = live
        count += 1
    # Post-condition: every staged scale is back on the storage the CUDA graph captured. A module
    # that was staged but not visited above (predicate changed between stage and reprocess) would
    # otherwise keep serving the staging buffer while the graph replays the stale live one.
    for module, sname, live in staged:
        param = getattr(module, sname)
        if param.data.data_ptr() != live.data_ptr():
            raise RuntimeError(
                f"{type(module).__name__}.{sname}: staged for the sync but not folded back into its live "
                "storage by reprocess_mxfp8_moe_layers; the CUDA graph would keep reading stale scales."
            )
    return count
