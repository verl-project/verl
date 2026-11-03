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

from collections.abc import Iterable

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
            # 0xFF is NaN-like for UE8M0 (2**128): a scale the sync fails to write is reported by
            # _check_moe_scales_written instead of being swizzled into the live buffer.
            staging.fill_(_UNWRITTEN_SCALE)
            pending.append((module, sname, param, staging))
    # Rebind only once every staging buffer is allocated: a failed allocation must not leave a
    # parameter on a staging buffer that no returned list records.
    staged = []
    for module, sname, param, staging in pending:
        staged.append((module, sname, param.data))
        param.data = staging
    return staged


_UNWRITTEN_SCALE = 0xFF


def _check_moe_scales_written(name: str, module: torch.nn.Module, staged_names: Iterable[str]) -> None:
    """Report a staged expert scale that still holds the staging sentinel after ``load_weights``."""
    from verl.utils.mxfp8_refit_check import report_refit_check_failure

    for sname in staged_names:
        data = getattr(module, sname).data
        unwritten = int((data == _UNWRITTEN_SCALE).sum().item())
        if unwritten == 0:
            continue
        report_refit_check_failure(
            f"{name}.{sname}: the weight sync did not write {unwritten} of {data.numel()} expert scale entries "
            "(they still hold the 0xFF staging sentinel). The sync shipped these experts without MXFP8 scales - "
            "typically their HF names miss the sync-side quantization rule (verl/utils/fp8_utils.py, e.g. "
            "Mixtral's block_sparse_moe.experts.N.w1/w2/w3) while the engine built them as fp8, so the engine "
            "cast bf16 weights into the fp8 buffer and would pair them with NaN scales. Fix the sync rule or "
            "quantization_config.ignored_layers so both sides agree."
        )


def reprocess_mxfp8_moe_layers(model: torch.nn.Module, staged: list[tuple[torch.nn.Module, str, torch.Tensor]]) -> int:
    """Re-run the MoE post-load processing and fold results into the captured storage."""
    from verl.utils.mxfp8_refit_check import refit_check_enabled

    live_by_key = {(id(m), n): live for m, n, live in staged}
    check = refit_check_enabled()
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
        if check:
            _check_moe_scales_written(name, module, [n for _, n in _MOE_SCALE_PAIRS if (id(module), n) in live_by_key])
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


# ---------------------------------------------------------------------------
# MoE layers kept in BF16
# ---------------------------------------------------------------------------

_BF16_MOE_CHECKED_ATTR = "_verl_sglang_bf16_moe_layouts_checked"
_MOE_DIMS = ("num_local_experts", "intermediate_size_per_partition", "hidden_size")


def _is_unquantized_moe(module: torch.nn.Module) -> bool:
    return isinstance(getattr(module, "w13_weight", None), torch.Tensor) and (
        type(getattr(module, "quant_method", None)).__name__ == "UnquantizedFusedMoEMethod"
    )


def relaid_bf16_moe_layers(model: torch.nn.Module) -> list[str]:
    """MoE layers kept in BF16 whose runner re-laid out the experts after loading, in a form a sync cannot write.

    ``ignored_layers`` can keep MoE layers in BF16 (``UnquantizedFusedMoEMethod``). On the DeepGEMM MoE runner,
    verl's default for quantized rollouts, they keep the checkpoint layout. On sglang's FlashInfer TRT-LLM
    runner (``engine_kwargs.sglang.moe_runner_backend=flashinfer_trtllm``), ``process_weights_after_loading``
    reorders their w3 / w1 rows and converts w13 / w2 into the runner's block layout (sglang 0.5.20), while
    ``FusedMoE.weight_loader`` copies each refit in checkpoint layout: the copy fails on the changed shape or,
    where the shape survives, writes rows the kernel reads in another order.
    Flags the TRT-LLM runner, and on other runners a w13 that no longer has its checkpoint shape
    ``[experts, 2 * intermediate, hidden]``.
    """
    found = []
    for name, module in model.named_modules():
        if not _is_unquantized_moe(module):
            continue
        w13 = module.w13_weight
        trtllm = any(getattr(obj, "use_flashinfer_trtllm_moe", False) for obj in (module.quant_method, module))
        dims = [getattr(module, attr, None) for attr in _MOE_DIMS]
        gated = getattr(getattr(module, "moe_runner_config", None), "is_gated", True)
        canonical = None if None in dims else (dims[0], (2 if gated else 1) * dims[1], dims[2])
        if trtllm or (canonical is not None and tuple(w13.shape) != canonical):
            runner = "FlashInfer TRT-LLM runner" if trtllm else f"checkpoint layout {canonical}"
            found.append(f"{name} (w13 {tuple(w13.shape)} {str(w13.dtype).removeprefix('torch.')}, {runner})")
    return found


def refuse_relaid_bf16_moe_layers(model: torch.nn.Module) -> None:
    """Raise before the first sync writes anything if a BF16 MoE layer's experts were re-laid out at load."""
    if model.__dict__.get(_BF16_MOE_CHECKED_ATTR):
        return
    found = relaid_bf16_moe_layers(model)
    if found:
        raise NotImplementedError(
            f"SGLang serves {len(found)} MoE layer(s) in BF16 whose experts the MoE runner re-laid out after "
            f"loading: {'; '.join(found[:4])}{'; ...' if len(found) > 4 else ''}. A weight sync writes "
            "checkpoint-layout tensors, which would not fit or would land in the wrong rows (the sglang limitation "
            "behind verl issue #8229). Keep verl's default DeepGEMM MoE runner (do not override "
            "engine_kwargs.sglang.moe_runner_backend), which leaves BF16 experts in checkpoint layout, or keep these "
            "MoE layers quantized."
        )
    model.__dict__[_BF16_MOE_CHECKED_ATTR] = True
