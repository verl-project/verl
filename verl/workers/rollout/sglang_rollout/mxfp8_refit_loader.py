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
"""MXFP8 weight-sync loader for SGLang, loaded via ``--custom-weight-loader``.

Why this exists. SGLang's FlashInfer MXFP8 GEMM backends do not consume the
canonical ``weight_scale_inv`` (UE8M0, ``[n, k/32]``) directly: at load time
``Fp8LinearMethod._process_mxfp8_linear_weight_scale`` derives a kernel layout
from it — ``flashinfer_cutlass`` stores an interleaved copy in
``weight_scale_inv_swizzled`` and ``apply()`` reads only that copy. A weight
sync overwrites ``weight`` / ``weight_scale_inv`` through ``model.load_weights``
and nothing re-derives the copy, so after the first sync the kernel pairs fresh
weights with stale scales and generation is garbage (observed on 2xB200,
SGLang 0.5.12: held-out acc 0.0, kl 7.2, entropy ~ log(vocab), from step 0).

What it does. ``load_and_reprocess`` is registered as a custom loader and named
as the request ``load_format``; SGLang ``dynamic_import``s it inside every TP
worker and calls it with ``(model, named_tensors)``. It loads the tensors the
standard way, then re-runs ``process_weights_after_loading`` on every module
whose quant method is MXFP8, which rebuilds the kernel-layout copies from the
just written canonical scales.

Which backend a layer runs on is read from the layer, not from the launch flag.
sglang >= 0.5.18 (#33208) resolves the MXFP8 dense backend at construction and
stores it as ``quant_method.mxfp8_dense_backend`` — with no ``--fp8-gemm-backend``
that is FlashInfer CuTe-DSL or CUTLASS on Blackwell and DeepGEMM on Hopper, all
of which keep derived copies (``weight_scale_inv_swizzled`` /
``weight_scale_inv_deepgemm``) that a sync must rebuild. sglang <= 0.5.17 has no
such attribute and dispatches on the requested backend, where ``auto`` means
Triton, which reads the canonical scale directly and needs no re-processing.
The ``flashinfer_trtllm`` backend is rejected on both: it shuffles the *weight*
tensor itself in place at load, so re-running the processing on an
already-shuffled layer would shuffle twice; supporting it needs per-layer
tracking of which parameters a sync touched.

MoE experts. SGLang's ``Fp8MoEMethod`` (fp8-serialized MXFP8 checkpoint) rewrites
the expert scales ``w13_weight_scale_inv`` / ``w2_weight_scale_inv`` *in place*
at load: the default Triton MoE runner swizzles them (shape may change), DeepGEMM
packs them (dtype changes), CUTLASS and TRT-LLM keep them canonical. A sync must
therefore write canonical ``[E, N, K/32]`` uint8 scales into a canonical-shaped
buffer and then re-run the processing, folding the result back into the storage
the CUDA graph captured. That is ``stage_mxfp8_moe_scales`` before ``load_weights``
and ``reprocess_mxfp8_moe_layers`` after it — the SGLang analogue of verl's vLLM
pristine-layout cycle, without a record step because the canonical layout is
derivable from the expert weight shape.

Sync-vs-engine check. Before anything is written, every incoming linear weight's
dtype is compared with the dtype of the engine parameter that will receive it
(``check_sync_matches_engine``): the engine decided at build time which layers are
fp8, the sync decided by name, and where the two differ ``load_weights`` would
silently cast into the wrong buffer. Together with the trainer-side layer audit
(training vs the sync rule) this closes the chain training = sync = engine.

Self-check. After re-processing, one dense MXFP8 layer's own ``apply`` is compared
against a dequantized reference (``verl.utils.mxfp8_refit_check``); a stale or
mis-laid-out scale shows up as O(1) relative error and raises. One local expert
of the smallest MXFP8 MoE layer is probed the same way: its canonical ``w13`` /
``w2`` weights and scales are snapshotted after ``load_weights`` (before the
runner rewrites the scales), then every probe row is routed to that expert via a
``StandardTopKOutput`` and the layer's forward is compared against the gated-MLP
reference. Skipped when expert parallelism is on (routing to a global expert id
would need the EP dispatch) or when TP>1 without ``reduce_results``. For staged
MoE scales the loader also verifies, before re-processing, that the sync wrote every
entry: the staging buffer is pre-filled with ``0xFF`` (the UE8M0 NaN code), so an
expert whose HF name misses the sync-side quantization rule — shipped as bf16,
which the engine's weight loader silently casts into the fp8 buffer with no scale —
is reported by name instead of being swizzled into the live buffer. Both checks
are disabled by ``VERL_MXFP8_REFIT_CHECK=0``.

Register at server launch (verl does this when ``rollout.quantization=mxfp8``)::

    custom_weight_loader=["verl.workers.rollout.sglang_rollout.mxfp8_refit_loader.load_and_reprocess"]

and send every ``update_weights_from_tensor`` request with
``load_format`` set to the same import path.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)

MXFP8_BLOCK_SIZE = 32

# Import path callers pass as both --custom-weight-loader and load_format.
LOADER_FQN = "verl.workers.rollout.sglang_rollout.mxfp8_refit_loader.load_and_reprocess"


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
        qm.process_weights_after_loading(module)
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
    buffer is no longer in checkpoint layout; ``reprocess_mxfp8_moe_layers`` folds
    the re-derived layout back into ``live_data``. Scales still in canonical layout
    (CUTLASS / TRT-LLM MoE runners) load in place and are not staged.
    """
    staged = []
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
            live = param.data
            staging = torch.empty(canonical, dtype=torch.uint8, device=live.device)
            # 0xFF is NaN-like for UE8M0 (2**128): a scale the sync fails to write blows
            # up in the self-check instead of being swizzled into the live buffer.
            staging.fill_(0xFF)
            param.data = staging
            staged.append((module, sname, live))
    return staged


_UNWRITTEN_SCALE = 0xFF


def _check_moe_scales_written(name: str, module: torch.nn.Module, staged_names: Iterable[str]) -> None:
    """Raise if a staged expert scale still holds the staging sentinel after ``load_weights``."""
    for sname in staged_names:
        data = getattr(module, sname).data
        unwritten = int((data == _UNWRITTEN_SCALE).sum().item())
        if unwritten == 0:
            continue
        raise RuntimeError(
            f"{name}.{sname}: the weight sync did not write {unwritten} of {data.numel()} expert scale entries "
            "(they still hold the 0xFF staging sentinel). The sync shipped these experts without MXFP8 scales - "
            "typically their HF names miss the sync-side quantization rule (verl/utils/fp8_utils.py, e.g. "
            "Mixtral's block_sparse_moe.experts.N.w1/w2/w3) while the engine built them as fp8, so the engine "
            "cast bf16 weights into the fp8 buffer and would pair them with NaN scales. Fix the sync rule or "
            "quantization_config.ignored_layers so both sides agree; VERL_MXFP8_REFIT_CHECK=0 bypasses this check."
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
# Self-check
# ---------------------------------------------------------------------------


def self_check_mxfp8_linear(model: torch.nn.Module) -> None:
    """Probe the smallest dense MXFP8 layer after re-processing (see ``mxfp8_refit_check``)."""
    from verl.utils.mxfp8_refit_check import assert_mxfp8_linear_matches, refit_check_enabled

    if not refit_check_enabled():
        return
    candidate = None
    for name, module in model.named_modules():
        if not _is_mxfp8_linear(module) or not hasattr(module, "weight"):
            continue
        if candidate is None or module.weight.numel() < candidate[1].weight.numel():
            candidate = (name, module)
    if candidate is None:
        return
    name, module = candidate
    qm = module.quant_method
    assert_mxfp8_linear_matches(
        name,
        lambda x: qm.apply(module, x),
        module.weight.data,
        module.weight_scale_inv.data,
        engine="sglang",
    )


def snapshot_mxfp8_moe_for_check(model: torch.nn.Module):
    """Pick the smallest MXFP8 MoE layer and copy one local expert's canonical weights / scales.

    Must run after ``load_weights`` and before ``reprocess_mxfp8_moe_layers``: the
    scales are canonical ``[E, N, K/32]`` uint8 at that point (staging buffers for
    runners that rewrite them, the live buffers for runners that keep them). Returns
    ``None`` when the check is disabled, no MoE layer qualifies, or the layer's
    parallelism is one the probe cannot reproduce.
    """
    from verl.utils.mxfp8_refit_check import refit_check_enabled

    if not refit_check_enabled():
        return None
    best = None
    for name, module in model.named_modules():
        if not _is_mxfp8_moe(module) or not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
            continue
        if best is None or module.w13_weight.numel() < best[1].w13_weight.numel():
            best = (name, module)
    if best is None:
        return None
    name, module = best
    if getattr(module, "moe_ep_size", 1) > 1:
        logger.debug("mxfp8 MoE refit check skipped on %s: expert parallelism %d", name, module.moe_ep_size)
        return None
    if getattr(module, "moe_tp_size", 1) > 1 and not getattr(module, "reduce_results", False):
        logger.debug("mxfp8 MoE refit check skipped on %s: TP without reduce_results", name)
        return None
    w13, w2 = module.w13_weight.data, module.w2_weight.data
    s13, s2 = module.w13_weight_scale_inv.data, module.w2_weight_scale_inv.data
    if (
        s13.dtype != torch.uint8
        or s2.dtype != torch.uint8
        or tuple(s13.shape) != _canonical_scale_shape(w13)
        or tuple(s2.shape) != _canonical_scale_shape(w2)
    ):
        logger.debug("mxfp8 MoE refit check skipped on %s: scales not in canonical layout after load", name)
        return None
    expert = 0  # local expert 0 == global expert 0 when EP is off
    return (
        name,
        module,
        expert,
        w13[expert].detach().clone(),
        s13[expert].detach().clone(),
        w2[expert].detach().clone(),
        s2[expert].detach().clone(),
    )


def _route_everything_to(module: torch.nn.Module, expert: int):
    """Return ``apply_fn(x)`` running the fused-MoE layer with every row routed to ``expert`` at weight 1."""

    def _apply(x: torch.Tensor) -> torch.Tensor:
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        rows = x.shape[0]
        topk_weights = torch.ones(rows, 1, dtype=torch.float32, device=x.device)
        topk_ids = torch.full((rows, 1), expert, dtype=torch.int32, device=x.device)
        num_experts = getattr(module, "num_experts", None) or module.w13_weight.shape[0]
        router_logits = torch.zeros(rows, num_experts, dtype=torch.float32, device=x.device)
        return module.forward(x, StandardTopKOutput(topk_weights, topk_ids, router_logits))

    return _apply


def self_check_mxfp8_moe(snapshot) -> None:
    """Probe the expert captured by ``snapshot_mxfp8_moe_for_check`` after re-processing."""
    from verl.utils.mxfp8_refit_check import assert_mxfp8_moe_expert_matches

    if snapshot is None:
        return
    name, module, expert, w13_q, s13, w2_q, s2 = snapshot
    reduce_ref = None
    if getattr(module, "moe_tp_size", 1) > 1:
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        reduce_ref = tensor_model_parallel_all_reduce  # the layer applies the same reduction to its output
    assert_mxfp8_moe_expert_matches(
        name,
        expert,
        _route_everything_to(module, expert),
        w13_q,
        s13,
        w2_q,
        s2,
        engine="sglang",
        reduce_ref=reduce_ref,
    )


# ---------------------------------------------------------------------------
# Sync-vs-engine dtype check
# ---------------------------------------------------------------------------

# HF leaf names that can be linear weights: ``.weight`` and the fused expert tensors transformers >= 5 writes.
_CHECKED_LEAVES = ("weight", "gate_up_proj", "down_proj")
# HF projections SGLang fuses into one module (the model's stacked_params_mapping, restated).
_STACKED = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
}
_W2_LEAVES = ("down_proj", "w2")
_SYNC_CACHE_ATTR = "_verl_sync_engine_fp8"


def resolve_engine_param(model: torch.nn.Module, hf_name: str) -> torch.Tensor | None:
    """Map an HF parameter name onto the engine tensor that will receive it, or None if unknown.

    Walks the module tree by name, trying the fused module name for projections SGLang stacks
    (``q_proj`` -> ``qkv_proj``, ``gate_proj`` -> ``gate_up_proj``) and stopping at a fused-MoE block
    (a module holding ``w13_weight``), where the remaining per-expert parts select ``w13`` or ``w2``.
    """
    parts = hf_name.split(".")
    module_path = parts[:-1] if parts[-1] == "weight" else parts
    cur: object = model
    for i, part in enumerate(module_path):
        if hasattr(cur, "w13_weight"):
            rest = module_path[i:] + ([parts[-1]] if parts[-1] != "weight" else [])
            return cur.w2_weight if any(r in _W2_LEAVES for r in rest) else cur.w13_weight
        nxt = getattr(cur, part, None)
        if nxt is None and part in _STACKED:
            nxt = getattr(cur, _STACKED[part], None)
        if nxt is None:
            return None
        cur = nxt
    if hasattr(cur, "w13_weight"):
        return cur.w2_weight if parts[-1] in _W2_LEAVES else cur.w13_weight
    if isinstance(cur, torch.nn.Module):
        w = getattr(cur, "weight", None)
        return w if isinstance(w, torch.Tensor) else None
    return cur if isinstance(cur, torch.Tensor) else None


def check_sync_matches_engine(model: torch.nn.Module, named_tensors: list[tuple[str, torch.Tensor]]) -> int:
    """Raise if the sync ships a weight in a precision other than the engine parameter that receives it.

    The engine decided at build time which layers are fp8 (``ignored_layers`` and model code); the sync
    decided by name what to quantize. Where they differ, ``load_weights`` would silently ``copy_`` a bf16
    tensor into an fp8 buffer (no scale ever arrives) or fp8 data into a bf16 parameter. This is the
    "sync vs engine" half of the train/rollout layer audit, done where the engine can be seen. Returns
    the number of parameters checked. Disabled with ``VERL_MXFP8_REFIT_CHECK=0``.
    """
    from verl.utils.mxfp8_refit_check import refit_check_enabled

    if not refit_check_enabled():
        return 0
    cache: dict[str, bool | None] = model.__dict__.setdefault(_SYNC_CACHE_ATTR, {})
    problems = []
    checked = 0
    unresolved = 0
    for name, tensor in named_tensors:
        if name.rsplit(".", 1)[-1] not in _CHECKED_LEAVES:
            continue
        if name not in cache:
            target = resolve_engine_param(model, name)
            cache[name] = None if target is None else target.dtype == torch.float8_e4m3fn
        engine_fp8 = cache[name]
        if engine_fp8 is None:
            unresolved += 1
            continue
        checked += 1
        sync_fp8 = tensor.dtype == torch.float8_e4m3fn
        if sync_fp8 != engine_fp8:
            problems.append(
                f"{name}: the sync ships {'fp8' if sync_fp8 else str(tensor.dtype)}, the engine parameter is "
                f"{'fp8' if engine_fp8 else 'high precision'}"
            )
    if unresolved:
        logger.debug(
            "sync-vs-engine dtype check: %d parameter names could not be mapped onto engine modules", unresolved
        )
    if problems:
        raise RuntimeError(
            f"MXFP8 weight sync and the SGLang engine disagree on the precision of {len(problems)} parameter(s):\n  - "
            + "\n  - ".join(problems[:40])
            + ("\n  - ..." if len(problems) > 40 else "")
            + "\nThe engine built these layers by quantization_config.ignored_layers and model code; the sync decided "
            "by the name rule in verl/utils/fp8_utils.py. load_weights would cast the tensor into the mismatched "
            "buffer silently (an fp8 layer never receives its scale). Make the sync rule and ignored_layers agree "
            "for these names. VERL_MXFP8_REFIT_CHECK=0 bypasses this check (not recommended)."
        )
    return checked


def load_and_reprocess(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Standard ``load_weights`` bracketed by MXFP8 layout staging / re-processing, then self-checks."""
    named_tensors = list(named_tensors)
    check_sync_matches_engine(model, named_tensors)  # before anything is written into the engine
    staged = stage_mxfp8_moe_scales(model)
    model.load_weights(named_tensors)
    moe_snapshot = snapshot_mxfp8_moe_for_check(model)
    n = reprocess_mxfp8_layers(model)
    m = reprocess_mxfp8_moe_layers(model, staged)
    if n or m:
        logger.debug("mxfp8 refit: re-derived kernel scale layouts for %d linear and %d MoE modules", n, m)
    self_check_mxfp8_linear(model)
    self_check_mxfp8_moe(moe_snapshot)
