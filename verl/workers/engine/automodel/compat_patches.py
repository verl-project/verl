# Copyright 2024 Bytedance Ltd. and/or its affiliates.
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

"""Runtime compatibility patches for nemo_automodel.

Fixes for nemo_automodel issues that upstream has not merged yet are applied
here as runtime monkey-patches, in the same spirit as the sglang
``weight_sync_patches`` module. Every patch is fail-closed: if the upstream
code it targets changes shape, the patch logs a warning and leaves upstream
behavior untouched. All patches are idempotent and self-disable once upstream
provides the equivalent behavior.

Patches
-------
1. ``state_dict_utils.get_expert_slice_for_rank``: the EP-free (plain-FSDP2)
   uneven expert-shard path computes per-rank expert ranges with a
   remainder-first formula, but PyTorch ``Shard.local_shard_size_and_offset``
   (what FSDP2 actually uses) distributes uneven shards with a fixed
   ``ceil(n/w)`` stride. They disagree on non-power-of-two worlds and weight
   loading fails with a shape mismatch. The patch recomputes the range with
   the PyTorch stride semantics; on evenly divisible worlds both formulas are
   identical.

2. DeepSeek-V4 FP8/FP4 checkpoint-load placeholder global shapes: the adapter
   builds destination tensors for the DCP plan via ``DTensor.from_local``
   without ``shape=``, which infers an EVEN global shape (local x mesh_size)
   and breaks the DCP load on plain-FSDP2 uneven dim-0 shards. The patch
   re-wraps the rebuilt placeholders with the true global shape/stride.

3. ``deepseek_v3.state_dict_adapter.dequantize_from_fp8``: same even-sharding
   inference bug on the dequantized output, plus the local rows of an uneven
   dim-0 shard are not 128-block aligned, so the dequant kernels silently use
   wrong scale rows past block boundaries. The patch re-wraps with the true
   global shape and pads the local rows to whole global blocks. Rebound in
   every module that from-imported it (deepseek_v4, mistral4 adapters).

4. ``transformers.utils.hub.resolve_revision`` shim for transformers < 5.18:
   automodel imports the name at module import time; older transformers
   lacks it and every worker dies with ImportError. The shim mirrors the
   5.18 semantics (local paths never reach it; hub paths resolve via
   ``HfApi.resolve_revision`` with a revision passthrough fallback). Must run
   before any ``nemo_automodel`` import, which is why it is applied first.

5. ``DeepSeekV4StateDictAdapter.to_hf_iter`` streaming export: verl streams
   the weight-sync export tensor-by-tensor; upstream only offers the
   full-dict ``to_hf``. The patch attaches a streaming variant that applies
   the internal-side hash-gate-bias drop once and converts one tensor at a
   time (per-tensor ``full_tensor()`` all-gathers stay bounded).

6. ``DeepseekV4ModelParallelizer`` whole-layer activation checkpointing: the
   DSV4 HyperConnection residual stream is [B, S, hc_mult=4, hidden] and the
   HC combine math lives outside the self_attn/mlp submodule
   CheckpointWrappers, so submodule-granularity activation checkpointing
   retains ~6 [S, 4, H] tensors per layer and OOMs the update step on long
   sequences. The patch forces one whole-layer AC boundary per layer (the
   same override gemma4_moe ships upstream).

7. ``dsv4_sinkhorn_normalize`` graceful torch fallback: ``attn='tilelang'``
   requests all DSV4 kernels together, but the sinkhorn kernels live in the
   separate ``tile_kernels`` package which may be missing while tilelang
   itself is installed. The patch degrades only the sinkhorn call to the
   torch reference (warns once) instead of raising.

8. ``DeepseekV4ForCausalLM.from_config`` ``compress_ratios`` compat:
   transformers >= 5.x ships its own ``deepseek_v4`` config that replaced the
   per-layer ``compress_ratios`` list with ``layer_types`` +
   ``compress_rates``. The patch rebuilds the list in place (idempotent)
   before the model is constructed.

9. ``DeepSeekV4StateDictAdapter`` plain-FSDP2 (EP=1) uneven expert-shard
   state_dict handling: with EP disabled each rank only sees the experts it
   owns (its uneven ``torch.chunk`` shard), but ``_aggregate_experts``
   assumes the full expert range per rank and ``create_dtensor_from_local``
   infers the wrong global shape. The patch stashes the merged expert
   tensors' DTensor specs during to_hf, infers per-layer expected counts
   from the keys present, re-wraps rank-local stacks with the stashed spec,
   and materializes the non-contiguous ``.T`` views (streaming weight sync
   calls ``.view(-1)`` on every tensor).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re

import torch

logger = logging.getLogger(__name__)

_applied: set[str] = set()


def _pytorch_shard_bounds(total: int, num_chunks: int, rank: int) -> tuple[int, int]:
    """Replicate ``Shard.local_shard_size_and_offset`` stride semantics.

    Returns the (start, end) range of ``total`` elements owned by ``rank``
    when the dim is sharded across ``num_chunks`` ranks: fixed stride
    ``ceil(total / num_chunks)``, last ranks may get fewer elements, and
    ranks beyond ``total`` get an empty range.
    """
    if total % num_chunks == 0:
        size = total // num_chunks
        start = size * rank
        return start, start + size
    full = (total + num_chunks - 1) // num_chunks
    start = full * rank
    if total <= start:
        return start, start
    return start, min(total, start + full)


def _source_contains(obj, marker: str) -> bool:
    """True when the upstream source of ``obj`` still contains ``marker``."""
    try:
        return marker in inspect.getsource(obj)
    except (OSError, TypeError):
        return False


def _contiguous_strides(shape) -> tuple[int, ...]:
    strides = []
    acc = 1
    for dim in reversed(tuple(shape)):
        strides.append(acc)
        acc *= dim
    return tuple(reversed(strides))


# ---------------------------------------------------------------------------
# Patch 1: EP-free uneven expert-shard ranges (PyTorch stride semantics)
# ---------------------------------------------------------------------------


def _patch_expert_slice_for_rank() -> bool:
    """Fix EP-free uneven expert-shard ranges on non-power-of-two worlds."""
    try:
        from nemo_automodel.components.moe import state_dict_utils as sdu
        from torch.distributed._tensor.placement_types import Replicate, Shard
    except Exception as e:  # pragma: no cover - image without nemo_automodel
        logger.warning("automodel compat: nemo_automodel not importable, skipping expert-shard patch: %s", e)
        return False

    def get_expert_slice_for_rank(experts_tensor, n_experts: int):
        """Patched: PyTorch stride semantics for uneven Shard(0) expert dims.

        Mirrors upstream ``get_expert_slice_for_rank`` except that the
        per-rank expert range on a ``Shard(0)`` mesh dimension follows
        ``Shard.local_shard_size_and_offset`` (fixed ceil stride) instead of
        the remainder-first formula. Empty shards (ranks past the tensor
        end) return an empty range, matching their empty local tensor.
        """
        if not sdu.is_dtensor(experts_tensor):
            return experts_tensor, 0, n_experts

        dtensor = experts_tensor
        local_tensor = dtensor.to_local()

        device_mesh = dtensor.device_mesh
        expert_mesh_dim_idx = sdu._get_expert_mesh_dim_index(dtensor)
        if expert_mesh_dim_idx is None:
            return local_tensor, 0, n_experts

        expert_mesh_dim_name = device_mesh.mesh_dim_names[expert_mesh_dim_idx]
        expert_mesh = sdu.get_submesh(device_mesh, (expert_mesh_dim_name,))
        placement = dtensor.placements[expert_mesh_dim_idx]
        if isinstance(placement, Shard) and placement.dim == 0:
            current_rank = expert_mesh.get_local_rank()
            world_size = expert_mesh.size()
            start_expert, end_expert = _pytorch_shard_bounds(n_experts, world_size, current_rank)
            return local_tensor, start_expert, end_expert
        if isinstance(placement, Replicate):
            return local_tensor, 0, n_experts
        return local_tensor, 0, n_experts

    try:
        sdu.get_expert_slice_for_rank = get_expert_slice_for_rank
        # split_experts_weights_dtensor_aware resolves the name from module
        # globals at call time, so replacing the module attribute suffices;
        # deepseek_v4's state_dict_adapter only from-imports the *split*
        # helper, which keeps calling into the patched module function.
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: failed to apply expert-shard patch: %s", e)
        return False
    return True


# ---------------------------------------------------------------------------
# Patch 2: DeepSeek-V4 FP8/FP4 placeholder global shapes (uneven FSDP2 shards)
# ---------------------------------------------------------------------------


def _fix_rebuilt_placeholder_global_shape(value, tensor, dtensor_cls):
    """Re-wrap a placeholder DTensor rebuilt from ``tensor`` with the true
    global shape when ``from_local``'s even-sharding inference got it wrong.

    Returns ``(value, fixed)``. Only touches tensors that share the source
    parameter's mesh, placements and LOCAL shape but disagree on the global
    shape - exactly the signature of the buggy ``from_local`` inference.
    """
    try:
        if not isinstance(value, dtensor_cls) or not isinstance(tensor, dtensor_cls):
            return value, False
        if value.device_mesh is not tensor.device_mesh:
            return value, False
        if tuple(value.placements) != tuple(tensor.placements):
            return value, False
        if tuple(value.shape) == tuple(tensor.shape):
            return value, False
        value_local = value.to_local()
        tensor_local = tensor.to_local()
        if tuple(value_local.shape) != tuple(tensor_local.shape):
            return value, False
        fixed = dtensor_cls.from_local(
            value_local,
            value.device_mesh,
            value.placements,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
        )
        return fixed, True
    except Exception as e:  # pragma: no cover - never corrupt the load path
        logger.warning(
            "automodel compat: placeholder global-shape fixup failed for a tensor (%s); keeping upstream value", e
        )
        return value, False


def _patch_dsv4_placeholder_global_shape() -> bool:
    """Fix DTensor placeholder rebuilds in the DeepSeek-V4 state dict adapter."""
    try:
        dsv4_mod = importlib.import_module("nemo_automodel.components.models.deepseek_v4.state_dict_adapter")
        cls = dsv4_mod.DeepSeekV4StateDictAdapter
        dtensor_cls = dsv4_mod.DTensor
    except Exception as e:  # pragma: no cover
        logger.warning(
            "automodel compat: deepseek_v4.state_dict_adapter not importable, skipping placeholder patch: %s", e
        )
        return False

    applied_any = False

    # 2a. Wrap convert_single_tensor_to_hf: fix any rebuilt destination
    # DTensor that shares the source parameter's local shard but carries an
    # inferred (wrong) global shape. Covers the non-expert FP8 placeholder
    # (DCP plan fails on e.g. layers.0.attn.wkv.weight: saved [512, 4096] vs
    # current [528, 4096] on 48 ranks) and stays a no-op for exports.
    orig_convert = cls.convert_single_tensor_to_hf
    if _source_contains(orig_convert, "fp8_val = DTensor.from_local(local_fp8, value.device_mesh, value.placements)"):

        def convert_single_tensor_to_hf(self, fqn, tensor, **kwargs):
            result = orig_convert(self, fqn, tensor, **kwargs)
            if not isinstance(tensor, dtensor_cls) or not result:
                return result
            fixed_result = []
            n_fixed = 0
            for key, value in result:
                fixed_value, changed = _fix_rebuilt_placeholder_global_shape(value, tensor, dtensor_cls)
                n_fixed += changed
                fixed_result.append((key, fixed_value))
            if n_fixed:
                logger.debug(
                    "automodel compat: re-wrapped %d placeholder DTensor(s) from %s with the true global shape",
                    n_fixed,
                    fqn,
                )
            return fixed_result

        try:
            cls.convert_single_tensor_to_hf = convert_single_tensor_to_hf
            applied_any = True
        except Exception as e:  # pragma: no cover
            logger.warning("automodel compat: failed to patch convert_single_tensor_to_hf: %s", e)
    else:
        logger.warning(
            "automodel compat: convert_single_tensor_to_hf drifted from upstream; "
            "FP8 placeholder global-shape fix skipped"
        )

    # 2b. _build_fp8_expert_placeholders: pass the true global shape/stride.
    orig_fp8 = cls._build_fp8_expert_placeholders
    if _source_contains(orig_fp8, "fp8_val = DTensor.from_local(local_fp8, value.device_mesh, value.placements)"):

        def _build_fp8_expert_placeholders(value):
            if dsv4_mod.is_dtensor(value):
                local_fp8 = cls._empty_or_cast_fp8(value.to_local())
                fp8_val = dtensor_cls.from_local(
                    local_fp8,
                    value.device_mesh,
                    value.placements,
                    shape=tuple(value.shape),
                    stride=tuple(value.stride()),
                )
            else:
                fp8_val = cls._empty_or_cast_fp8(value)
            scale = cls._build_fp8_dtensor_scale_placeholder(value)
            return fp8_val, scale

        try:
            cls._build_fp8_expert_placeholders = staticmethod(_build_fp8_expert_placeholders)
            applied_any = True
        except Exception as e:  # pragma: no cover
            logger.warning("automodel compat: failed to patch _build_fp8_expert_placeholders: %s", e)
    else:
        logger.warning(
            "automodel compat: _build_fp8_expert_placeholders drifted from upstream; global-shape fix skipped"
        )

    # 2c. _build_fp4_expert_placeholders: the packed int8 / e8m0 scale global
    # shapes are derived from the source DTensor's global shape (in-dim / 2
    # and / FP4_COL_BLOCK), not from the inferred even-shard global.
    orig_fp4 = cls._build_fp4_expert_placeholders
    if _source_contains(orig_fp4, "packed_d = DTensor.from_local(packed, value.device_mesh, value.placements)"):
        fp4_col_block = dsv4_mod.FP4_COL_BLOCK

        def _build_fp4_expert_placeholders(value):
            if dsv4_mod.is_dtensor(value):
                local = value.to_local()
                in_dim = local.shape[-1]
                assert in_dim % fp4_col_block == 0, f"V4 expert in-dim {in_dim} must be divisible by {fp4_col_block}"
                packed = torch.empty(*local.shape[:-1], in_dim // 2, dtype=torch.int8, device=local.device)
                scale = torch.empty(
                    *local.shape[:-1],
                    in_dim // fp4_col_block,
                    dtype=torch.float8_e8m0fnu,
                    device=local.device,
                )
                global_in = value.shape[-1]
                assert global_in % fp4_col_block == 0, (
                    f"V4 expert global in-dim {global_in} must be divisible by {fp4_col_block}"
                )
                packed_gshape = tuple(value.shape[:-1]) + (global_in // 2,)
                scale_gshape = tuple(value.shape[:-1]) + (global_in // fp4_col_block,)
                packed_d = dtensor_cls.from_local(
                    packed,
                    value.device_mesh,
                    value.placements,
                    shape=packed_gshape,
                    stride=_contiguous_strides(packed_gshape),
                )
                scale_d = dtensor_cls.from_local(
                    scale,
                    value.device_mesh,
                    value.placements,
                    shape=scale_gshape,
                    stride=_contiguous_strides(scale_gshape),
                )
                return packed_d, scale_d

            in_dim = value.shape[-1]
            assert in_dim % fp4_col_block == 0, f"V4 expert in-dim {in_dim} must be divisible by {fp4_col_block}"
            packed = torch.empty(*value.shape[:-1], in_dim // 2, dtype=torch.int8)
            scale = torch.empty(*value.shape[:-1], in_dim // fp4_col_block, dtype=torch.float8_e8m0fnu)
            return packed, scale

        try:
            cls._build_fp4_expert_placeholders = staticmethod(_build_fp4_expert_placeholders)
            applied_any = True
        except Exception as e:  # pragma: no cover
            logger.warning("automodel compat: failed to patch _build_fp4_expert_placeholders: %s", e)
    else:
        logger.warning(
            "automodel compat: _build_fp4_expert_placeholders drifted from upstream; global-shape fix skipped"
        )

    return applied_any


# ---------------------------------------------------------------------------
# Patch 3: dequantize_from_fp8 for uneven FSDP2 shards
# ---------------------------------------------------------------------------


def _patch_dequantize_from_fp8() -> bool:
    """Fix the from_hf FP8 dequantization path for uneven dim-0 shards."""
    try:
        v3_mod = importlib.import_module("nemo_automodel.components.models.deepseek_v3.state_dict_adapter")
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: deepseek_v3.state_dict_adapter not importable, skipping dequant patch: %s", e)
        return False

    orig_dequant = v3_mod.dequantize_from_fp8
    if not _source_contains(
        orig_dequant, "return DTensor.from_local(dequantized_local, weight.device_mesh, weight.placements)"
    ):
        logger.warning("automodel compat: dequantize_from_fp8 drifted from upstream; uneven-shard dequant fix skipped")
        return False

    block_size = v3_mod.BLOCK_SIZE

    def dequantize_from_fp8(
        weight,
        scale_inv,
        dtype=torch.bfloat16,
        BLOCK_SIZE=block_size,
        name: str = "",
    ):
        """Patched dequantize_from_fp8.

        Differences vs upstream (only active for DTensor weights with a
        full-tensor scale):

        - The dequantized result is re-wrapped with the weight DTensor's
          TRUE global shape/stride instead of from_local's even-sharding
          inference (global = local x mesh_size).
        - The local row range is computed with torch.chunk semantics from
          the weight's true global shape, and the local rows are padded to
          whole global 128-blocks (matching the sliced scale) before the
          dequant kernel runs; the padding is cropped afterwards. Without
          this, rows past a global block boundary silently use the wrong
          scale factor.
        """
        from torch.distributed._tensor import DTensor, Shard

        is_dtensor = v3_mod.is_dtensor
        weight_is_dtensor = is_dtensor(weight)
        scale_is_dtensor = is_dtensor(scale_inv)

        weight_local = weight.to_local() if weight_is_dtensor else weight
        scale_local = scale_inv.to_local() if scale_is_dtensor else scale_inv

        expected_scale_shape = v3_mod.calculate_scale_shape(weight_local, BLOCK_SIZE)

        def _dispatch_dequant(w, s):
            use_triton = v3_mod._TRITON_AVAILABLE and w.is_cuda and s.is_cuda and w.dim() == 2 and s.dim() == 2
            if use_triton:
                try:
                    return v3_mod._dequantize_with_triton(w, s, dtype, BLOCK_SIZE)
                except Exception as exc:
                    logger.warning("Triton dequant failed (%s). Falling back to torch.", exc)
                    return v3_mod._dequantize_with_torch(w, s, dtype, BLOCK_SIZE)
            return v3_mod._dequantize_with_torch(w, s, dtype, BLOCK_SIZE)

        def _finish(local_dequant):
            if weight_is_dtensor:
                # Pass the true global shape/stride explicitly: from_local
                # assumes EVEN sharding and would infer global = local x
                # mesh_size, which is wrong for uneven dim-0 FSDP2 shards.
                return DTensor.from_local(
                    local_dequant,
                    weight.device_mesh,
                    weight.placements,
                    shape=tuple(weight.shape),
                    stride=tuple(weight.stride()),
                )
            return local_dequant

        if scale_local.shape != expected_scale_shape:
            logger.debug(
                "%s scale_inv shape %s doesn't match expected shape %s, slicing scale_inv",
                name,
                scale_local.shape,
                expected_scale_shape,
            )
            if weight_is_dtensor and not scale_is_dtensor:
                # Locate the Shard(0) mesh dim to compute this rank's exact
                # global row range (torch.chunk semantics on the true global
                # shape). Replicate the uneven-safe upstream slicing flow.
                shard_dim = None
                shard_mesh_dim = None
                for mesh_dim, placement in enumerate(weight.placements):
                    if isinstance(placement, Shard):
                        shard_dim = placement.dim
                        shard_mesh_dim = mesh_dim
                        break
                handled = False
                if shard_dim == 0:
                    mesh_size = weight.device_mesh.size(shard_mesh_dim)
                    mesh_coord = weight.device_mesh.get_local_rank(mesh_dim=shard_mesh_dim)
                    global_rows = weight.shape[0]
                    local_rows = weight_local.shape[0]
                    chunk = (global_rows + mesh_size - 1) // mesh_size
                    start_row = min(mesh_coord * chunk, global_rows)
                    end_row = start_row + local_rows
                    start_block = start_row // BLOCK_SIZE
                    end_block = min((end_row + BLOCK_SIZE - 1) // BLOCK_SIZE, scale_inv.shape[0])
                    scale_local = scale_inv[start_block:end_block, :].contiguous()
                    # Uneven dim-0 FSDP2 shards are NOT 128-block aligned:
                    # the local row range may start mid-block and/or cross
                    # block boundaries. The dequant kernels index scale rows
                    # as local_row // BLOCK_SIZE, which is only globally
                    # correct when the padded range starts exactly at a
                    # global block boundary. Pad the local rows to whole
                    # blocks (aligned with the sliced scale), dequantize,
                    # then trim.
                    top_pad = start_row % BLOCK_SIZE
                    padded_rows = (end_block - start_block) * BLOCK_SIZE
                    bottom_pad = padded_rows - top_pad - local_rows
                    if top_pad != 0 or bottom_pad != 0:
                        if bottom_pad < 0:
                            logger.warning(
                                "%s scale slice covers fewer rows (%d blocks) than the local weight needs "
                                "(offset %d + %d rows); falling back to unsliced dequant",
                                name,
                                end_block - start_block,
                                top_pad,
                                local_rows,
                            )
                        else:
                            padded = torch.nn.functional.pad(weight_local, (0, 0, top_pad, bottom_pad))
                            dequantized_padded = _dispatch_dequant(padded, scale_local.to(device=weight_local.device))
                            return _finish(dequantized_padded[top_pad : top_pad + local_rows])
                    handled = True
                if not handled:
                    # Shard on a non-zero dim: fall back to upstream slicing.
                    scale_local = v3_mod._slice_scale_for_dtensor(scale_inv, weight, weight_local, BLOCK_SIZE)
                # Verify the slice worked
                if scale_local.shape != expected_scale_shape:
                    logger.warning(
                        "%s scale_inv shape %s still doesn't match expected shape %s after slicing",
                        name,
                        scale_local.shape,
                        expected_scale_shape,
                    )

        scale_local = scale_local.to(device=weight_local.device)
        if not weight_local.is_contiguous():
            weight_local = weight_local.contiguous()
        if not scale_local.is_contiguous():
            scale_local = scale_local.contiguous()

        return _finish(_dispatch_dequant(weight_local, scale_local))

    # Patch the defining module and every module that from-imported the name.
    rebound = [v3_mod]
    for mod_name in (
        "nemo_automodel.components.models.deepseek_v4.state_dict_adapter",
        "nemo_automodel.components.models.mistral4.state_dict_adapter",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # pragma: no cover - optional dependents
            continue
        if hasattr(mod, "dequantize_from_fp8"):
            rebound.append(mod)
    try:
        for mod in rebound:
            mod.dequantize_from_fp8 = dequantize_from_fp8
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: failed to apply dequantize_from_fp8 patch: %s", e)
        return False
    return True


# ---------------------------------------------------------------------------
# Patch 4: transformers.utils.hub.resolve_revision shim (transformers < 5.18)
# ---------------------------------------------------------------------------


def _ensure_transformers_hub_resolve_revision() -> bool:
    """Inject ``resolve_revision`` into ``transformers.utils.hub`` if missing.

    Returns True when the shim was installed (transformers < 5.18), False when
    the native implementation is already present or injection failed (in
    which case importing nemo_automodel will surface the original ImportError
    rather than failing silently).

    MUST run before any ``nemo_automodel`` import: ``_transformers.auto_config``
    does ``from transformers.utils.hub import extract_commit_hash,
    resolve_revision`` at module import time.
    """
    try:
        import transformers.utils.hub as hub_mod
    except Exception as e:  # pragma: no cover - image without transformers
        logger.warning("automodel compat: transformers.utils.hub not importable, skipping resolve_revision shim: %s", e)
        return False

    if getattr(hub_mod, "resolve_revision", None) is not None:
        return False  # transformers >= 5.18 provides it natively

    def resolve_revision(
        repo_id,
        revision=None,
        *,
        cache_dir=None,
        token=None,
        local_files_only=False,
        **kwargs,
    ):
        """Shim of transformers>=5.18 ``transformers.utils.hub.resolve_revision``.

        Resolves a Hub revision (branch/tag/PR) to a commit hash, returning a
        ``ResolvedRevision`` or the resolved sha. Callers only exercise this
        for Hub sources: automodel's ``NeMoAutoConfig.resolve_revision``
        wrapper short-circuits local paths via ``os.path.exists`` first, and
        falls back to ``hf_hub_download`` + ``extract_commit_hash`` when the
        returned value is not a 40-char sha — so a best-effort return of the
        original revision remains correct end-to-end.
        """
        try:
            from huggingface_hub import HfApi

            return HfApi().resolve_revision(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
        except Exception as e:
            logger.debug(
                "automodel compat: HfApi.resolve_revision unavailable for %r (%s); "
                "returning the original revision for the caller's fallback",
                repo_id,
                e,
            )
            return revision

    try:
        hub_mod.resolve_revision = resolve_revision
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: failed to inject resolve_revision shim: %s", e)
        return False
    return True


# ---------------------------------------------------------------------------
# Patch 5: DeepSeekV4StateDictAdapter.to_hf_iter streaming export
# ---------------------------------------------------------------------------


def _patch_dsv4_to_hf_iter() -> bool:
    """Attach the streaming to_hf export used by the weight-sync path."""
    try:
        dsv4_mod = importlib.import_module("nemo_automodel.components.models.deepseek_v4.state_dict_adapter")
        cls = dsv4_mod.DeepSeekV4StateDictAdapter
    except Exception as e:  # pragma: no cover
        logger.warning(
            "automodel compat: deepseek_v4.state_dict_adapter not importable, skipping to_hf_iter patch: %s", e
        )
        return False

    if getattr(cls, "to_hf_iter", None) is not None:
        return False  # upstream gained a streaming export; nothing to do

    # Fail-closed: the streaming method composes these main-branch pieces.
    if not all(
        hasattr(cls, name) for name in ("convert_single_tensor_to_hf", "_drop_hash_layer_gate_bias")
    ) or not hasattr(dsv4_mod, "_HashBiasScope"):
        logger.warning(
            "automodel compat: deepseek_v4 adapter is missing convert_single_tensor_to_hf / "
            "_drop_hash_layer_gate_bias / _HashBiasScope; to_hf_iter patch skipped"
        )
        return False

    _scope_internal = dsv4_mod._HashBiasScope.INTERNAL

    def to_hf_iter(
        self,
        state_dict,
        exclude_key_regex=None,
        tensor_converter=None,
        **kwargs,
    ):
        """Streaming variant of ``to_hf`` (compat re-implementation).

        Yields ``(hf_key, hf_value)`` pairs one internal tensor at a time
        instead of materializing the full HF dict, so callers can export
        large models from a sharded state dict with bounded memory.

        ``tensor_converter`` is applied to each internal tensor BEFORE the
        conversion (e.g. ``DTensor.full_tensor`` to gather the full plain
        tensor) and must preserve shapes.

        The internal-side hash-gate-bias drop is applied once up front. The
        HF-side belt-and-suspenders re-drop from ``to_hf`` is skipped here:
        it only guards against DCP-loaded dicts that already contain
        HF-format keys, which cannot happen for a freshly materialized
        internal state dict.
        """
        state_dict = self._drop_hash_layer_gate_bias(state_dict, _scope_internal)

        for fqn, tensor in state_dict.items():
            if exclude_key_regex is not None and re.match(exclude_key_regex, fqn):
                continue
            if tensor_converter is not None:
                tensor = tensor_converter(tensor)
            yield from self.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs)

    try:
        cls.to_hf_iter = to_hf_iter
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: failed to attach to_hf_iter: %s", e)
        return False
    return True


# ---------------------------------------------------------------------------
# Patch 6: DSV4 whole-layer activation checkpointing
# ---------------------------------------------------------------------------


def _patch_dsv4_full_layer_activation_checkpointing() -> bool:
    """Force the whole-layer AC boundary for DeepSeek-V4 (HC combine math
    lives outside the submodule CheckpointWrappers; submodule-granularity AC
    retains ~129GB of [S, 4, H] residual tensors across 43 layers @ 10k
    tokens and OOMs the update step)."""
    try:
        par_mod = importlib.import_module("nemo_automodel.components.models.deepseek_v4.parallelization")
        from nemo_automodel.components.distributed import ModelParallelizer
    except Exception as e:  # pragma: no cover - image without nemo_automodel
        logger.warning(
            "automodel compat: deepseek_v4.parallelization not importable, skipping full-layer AC patch: %s", e
        )
        return False

    cls = par_mod.DeepseekV4ModelParallelizer

    # Fail-closed: the base parallelizer must still own the hook the override
    # targets, and upstream must not have gained its own override.
    if "_use_full_layer_activation_checkpointing" not in vars(ModelParallelizer):
        logger.warning(
            "automodel compat: ModelParallelizer no longer defines "
            "_use_full_layer_activation_checkpointing; full-layer AC patch skipped"
        )
        return False
    if "_use_full_layer_activation_checkpointing" in vars(cls):
        return False  # upstream gained its own override; nothing to do

    def _use_full_layer_activation_checkpointing(self, model) -> bool:
        """Use WHOLE-LAYER activation checkpointing for DSV4.

        The DSV4 HyperConnection residual stream is [B, S, hc_mult=4, hidden]
        and the HC combine math lives OUTSIDE the self_attn/mlp submodule
        CheckpointWrappers, so submodule-granularity AC retains ~6 [S, 4, H]
        tensors per layer and OOMs the update step. A single layer-level
        boundary retains one such tensor per layer instead.
        """
        return True

    try:
        cls._use_full_layer_activation_checkpointing = _use_full_layer_activation_checkpointing
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: failed to apply full-layer AC patch: %s", e)
        return False
    return True


# ---------------------------------------------------------------------------
# Patch 7: dsv4 sinkhorn graceful torch fallback (image without tile_kernels)
# ---------------------------------------------------------------------------


def _patch_dsv4_sinkhorn_torch_fallback() -> bool:
    """Degrade the DSV4 sinkhorn call to the torch reference when tile_kernels
    is missing instead of raising RuntimeError from ``_should_use_tilelang``."""
    try:
        kern_mod = importlib.import_module("nemo_automodel.components.models.deepseek_v4.optimized_kernels")
        orig = kern_mod.dsv4_sinkhorn_normalize
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: deepseek_v4.optimized_kernels not importable, skipping sinkhorn patch: %s", e)
        return False

    # Fail-closed: the upstream function must still gate through
    # _should_use_tilelang with the kernel-availability probe.
    if not _source_contains(orig, "if _should_use_tilelang("):
        logger.warning("automodel compat: dsv4_sinkhorn_normalize drifted from upstream; torch-fallback patch skipped")
        return False

    warned = [False]

    def dsv4_sinkhorn_normalize(x, *, backend, repeat, eps):
        """Patched dsv4_sinkhorn_normalize.

        ``backend='tilelang'`` requests all DSV4 kernels together, but the
        sinkhorn kernels live in the separate ``tile_kernels`` package which
        may be missing while tilelang itself is installed; degrade only the
        sinkhorn call to the torch reference (warn once) so sparse attention /
        indexer can still use their TileLang kernels.
        """
        if backend == "tilelang" and not kern_mod.is_dsv4_kernel_available("sinkhorn"):
            if not warned[0]:
                warned[0] = True
                logger.warning(
                    "automodel compat: dsv4 sinkhorn TileLang backend was requested, but tile_kernels is "
                    "unavailable; falling back to the torch sinkhorn implementation."
                )
            backend = "torch"
        return orig(x, backend=backend, repeat=repeat, eps=eps)

    # Rebind in the defining module and every module that from-imported it.
    rebound = [kern_mod]
    for mod_name in (
        "nemo_automodel.components.models.deepseek_v4.layers",
        "nemo_automodel.components.models.deepseek_v41.layers",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # pragma: no cover - optional dependents
            continue
        if getattr(mod, "dsv4_sinkhorn_normalize", None) is orig:
            rebound.append(mod)
    try:
        for mod in rebound:
            mod.dsv4_sinkhorn_normalize = dsv4_sinkhorn_normalize
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: failed to apply sinkhorn fallback patch: %s", e)
        return False
    return True


# ---------------------------------------------------------------------------
# Patch 8: DSV4 compress_ratios compat (transformers >= 5.x naming)
# ---------------------------------------------------------------------------


def _ensure_compress_ratios_compat(config) -> None:
    """Rebuild ``compress_ratios`` (list) from the upstream transformers naming.

    transformers >= 5.x ships its own ``deepseek_v4`` ``DeepseekV4Config``
    that replaced the per-layer ``compress_ratios`` list with:
      - ``layer_types``: per-layer attention type names, and
      - ``compress_rates``: {attention_type: compression_ratio} dict.
    Because ``model_type`` is already registered there, ``AutoConfig.
    from_pretrained`` builds the transformers config. Automodel's model code
    (layers.py, mtp.py) expects the original list; rebuild it in place
    (idempotent) so both layouts work.
    """
    if hasattr(config, "compress_ratios") and config.compress_ratios:
        return
    rates = getattr(config, "compress_rates", None)
    if rates is None:
        return
    if isinstance(rates, list | tuple):
        # Same layout, different name.
        config.compress_ratios = list(rates)
        return
    if isinstance(rates, dict):
        layer_types = getattr(config, "layer_types", None)
        if layer_types:
            # Rebuild the per-layer list from the type names. Unknown / full /
            # sliding layers carry no compressor -> ratio 0.
            config.compress_ratios = [rates.get(t, 0) for t in layer_types]
        else:
            # No per-layer info available; treat every layer as unspecified.
            config.compress_ratios = []


def _patch_dsv4_compress_ratios_compat() -> bool:
    """Normalize the transformers>=5.x config naming before model build."""
    try:
        model_mod = importlib.import_module("nemo_automodel.components.models.deepseek_v4.model")
        cls = model_mod.DeepseekV4ForCausalLM
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: deepseek_v4.model not importable, skipping compress_ratios patch: %s", e)
        return False

    orig_from_config = cls.from_config
    if _source_contains(orig_from_config, "ensure_compress_ratios_compat"):
        return False  # upstream gained the fix; nothing to do
    # Fail-closed: the upstream classmethod must still be the plain pass-through.
    if not _source_contains(orig_from_config, "return cls(config, moe_config, backend, **kwargs)"):
        logger.warning(
            "automodel compat: DeepseekV4ForCausalLM.from_config drifted from upstream; "
            "compress_ratios compat patch skipped"
        )
        return False

    @classmethod
    def from_config(cls, config, moe_config=None, backend=None, **kwargs):
        """Patched from_config: rebuild ``compress_ratios`` from the upstream
        transformers naming (``layer_types`` + ``compress_rates``) before the
        model (and its MTP blocks, which copy this config) are built."""
        _ensure_compress_ratios_compat(config)
        return orig_from_config.__func__(cls, config, moe_config, backend, **kwargs)

    try:
        cls.from_config = from_config
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: failed to apply compress_ratios patch: %s", e)
        return False
    return True


# ---------------------------------------------------------------------------
# Patch 9: DSV4 plain-FSDP2 (EP=1) uneven expert-shard state_dict handling
# ---------------------------------------------------------------------------


_MERGED_GATE_UP_PROJS_FQN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_and_up_projs$")
_MERGED_DOWN_PROJS_FQN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_projs$")
_EXPERT_WEIGHT_FQN = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")


def _patch_dsv4_plain_fsdp_expert_state_dict() -> bool:
    """Fix from_hf/to_hf for plain-FSDP2 (EP=1) uneven expert-shard state dicts."""
    try:
        dsv4_mod = importlib.import_module("nemo_automodel.components.models.deepseek_v4.state_dict_adapter")
        cls = dsv4_mod.DeepSeekV4StateDictAdapter
    except Exception as e:  # pragma: no cover
        logger.warning(
            "automodel compat: deepseek_v4.state_dict_adapter not importable, skipping EP=1 expert patch: %s", e
        )
        return False

    if "_wrap_local_moe_stack" in vars(cls):
        return False  # upstream gained the fix; nothing to do
    # Fail-closed: the aggregation still assumes one full expert range per rank.
    if not _source_contains(cls._aggregate_experts, "expected_per_rank = n_experts"):
        logger.warning(
            "automodel compat: DeepSeekV4StateDictAdapter._aggregate_experts drifted from upstream; "
            "EP=1 expert-shard patch skipped"
        )
        return False

    # 9a. Spec stash + streaming-friendly contiguity on the to_hf side.
    # NOTE: patch 2a wraps convert_single_tensor_to_hf FIRST (its drift guard
    # reads the pristine upstream source); this wrapper chains on top of it.
    orig_convert = cls.convert_single_tensor_to_hf

    def _maybe_stash_moe_split_spec(self, fqn, tensor) -> None:
        """Record the merged MoE expert tensor's DTensor spec for from_hf.

        Only stashes for plain (non-EP) meshes: EP meshes keep the same
        expert count on every rank and from_hf rebuilds them through
        ``create_dtensor_from_local`` with the ep mesh; plain FSDP2 shards
        the expert dim unevenly (torch.chunk semantics), so from_hf needs the
        original global shape to wrap the rank-local stack correctly.
        """
        if not dsv4_mod.is_dtensor(tensor):
            return
        mesh = tensor.device_mesh
        if any(dim in mesh.mesh_dim_names for dim in ("ep", "ep_shard", "ep_replicate")):
            return
        self.__dict__.setdefault("_moe_split_spec", {})[fqn] = (
            mesh,
            tensor.placements,
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )

    def convert_single_tensor_to_hf(self, fqn, tensor, **kwargs):
        """Patched convert_single_tensor_to_hf (plain-FSDP2 EP=1 expert shards).

        Stashes the merged expert tensor's DTensor spec before the split, and
        materializes the non-contiguous ``.T`` views of the exported
        per-expert weights (the streaming weight-sync path calls ``.view(-1)``
        on every tensor, which requires contiguity).
        """
        if isinstance(tensor, dsv4_mod.DTensor) and (
            _MERGED_GATE_UP_PROJS_FQN.match(fqn) or _MERGED_DOWN_PROJS_FQN.match(fqn)
        ):
            self._maybe_stash_moe_split_spec(fqn, tensor)
        result = orig_convert(self, fqn, tensor, **kwargs)
        if not result:
            return result
        return [
            (key, value.contiguous())
            if _EXPERT_WEIGHT_FQN.match(key) and hasattr(value, "is_contiguous") and not value.is_contiguous()
            else (key, value)
            for key, value in result
        ]

    def _wrap_local_moe_stack(self, stacked, native_key, device_mesh, rank):
        """Wrap a rank-local merged-expert stack for the final model load.

        EP path: the stack is a uniform chunk of the expert dim, so
        ``create_dtensor_from_local`` with the ep mesh infers the right global
        shape. Plain FSDP2 (EP=1): the expert dim is unevenly sharded
        (torch.chunk semantics) and ``device_mesh`` is None, so use the spec
        stashed by ``_maybe_stash_moe_split_spec`` during to_hf to rebuild a
        DTensor whose global shape matches the model's state_dict entry
        ([n_experts, ...] Shard(0) over the FSDP mesh).
        """
        spec = self.__dict__.get("_moe_split_spec", {}).get(native_key)
        if spec is not None:
            mesh, placements, gshape, gstride = spec
            return dsv4_mod.DTensor.from_local(stacked, mesh, placements, shape=gshape, stride=gstride)
        return dsv4_mod.create_dtensor_from_local(stacked, device_mesh, rank)

    # 9b. Replace the aggregation with the uneven-shard-aware version.
    def _aggregate_experts(self, state_dict, device_mesh):
        """Patched _aggregate_experts: plain-FSDP2 (EP=1) uneven expert shards.

        Same as upstream except that with EP disabled each rank only sees
        the experts it owns in its state dict (its uneven dim-0 shard of the
        merged tensor), so the per-layer expected count is inferred from the
        keys present instead of assuming the full expert range, and the
        rank-local stacks are wrapped through ``_wrap_local_moe_stack``.
        """
        if self.expert_storage_format == "mxfp4":
            return self._aggregate_experts_packed(state_dict, device_mesh)
        n_experts = self.moe_config.n_routed_experts

        if device_mesh is not None:
            start_expert, end_expert = dsv4_mod.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_per_rank = end_expert - start_expert
            rank = (
                dsv4_mod.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
            expected_per_layer = None
        else:
            start_expert, end_expert = 0, n_experts
            expected_per_rank = None
            rank = None
            per_layer_expert_ids: dict[str, set[int]] = {}
            for key in state_dict:
                m = dsv4_mod._EXPERT_PATTERN.match(key)
                if m is not None:
                    per_layer_expert_ids.setdefault(m.group(1), set()).add(int(m.group(2)))
            expected_per_layer = {layer: len(ids) for layer, ids in per_layer_expert_ids.items()}

        def _expected_count(layer_num: str) -> int:
            if expected_per_rank is not None:
                return expected_per_rank
            return expected_per_layer.get(layer_num, 0)

        by_layer: dict[str, dict] = {}
        out: dict = {}

        for key in list(state_dict.keys()):
            value = state_dict.pop(key)
            m = dsv4_mod._EXPERT_PATTERN.match(key)
            if m is None:
                out[key] = value
                continue

            layer_num, expert_num, which = m.group(1), int(m.group(2)), m.group(3)

            if not dsv4_mod.should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                continue

            if layer_num not in by_layer:
                by_layer[layer_num] = {"gate_and_up": {}, "down": {}}

            if which in ("w1", "w3"):
                if expert_num not in by_layer[layer_num]["gate_and_up"]:
                    by_layer[layer_num]["gate_and_up"][expert_num] = {}
                by_layer[layer_num]["gate_and_up"][expert_num][which] = value
            else:  # w2 = down_proj
                by_layer[layer_num]["down"][expert_num] = value

            gu_layer = by_layer[layer_num].get("gate_and_up")
            if gu_layer is not None:
                all_ready = len(gu_layer) == _expected_count(layer_num) and all(
                    isinstance(d, dict) and "w1" in d and "w3" in d for d in gu_layer.values()
                )
                if all_ready:
                    expert_ids = sorted(gu_layer.keys())
                    tensors = []
                    for eid in expert_ids:
                        gate_w = gu_layer[eid]["w1"]
                        up_w = gu_layer[eid]["w3"]
                        if dsv4_mod.is_dtensor(gate_w):
                            gate_w = gate_w.to_local()
                        if dsv4_mod.is_dtensor(up_w):
                            up_w = up_w.to_local()
                        tensors.append(torch.cat([gate_w.T, up_w.T], dim=-1))
                    stacked = torch.stack(tensors, dim=0).to(self.dtype)
                    native_key = f"model.layers.{layer_num}.mlp.experts.gate_and_up_projs"
                    out[native_key] = self._wrap_local_moe_stack(stacked, native_key, device_mesh, rank)
                    del by_layer[layer_num]["gate_and_up"]

            down_layer = by_layer[layer_num].get("down")
            if down_layer is not None and len(down_layer) == _expected_count(layer_num):
                expert_ids = sorted(down_layer.keys())
                tensors = []
                for eid in expert_ids:
                    w = down_layer[eid]
                    if dsv4_mod.is_dtensor(w):
                        w = w.to_local()
                    tensors.append(w.T)
                stacked = torch.stack(tensors, dim=0).to(self.dtype)
                native_key = f"model.layers.{layer_num}.mlp.experts.down_projs"
                out[native_key] = self._wrap_local_moe_stack(stacked, native_key, device_mesh, rank)
                del by_layer[layer_num]["down"]

        return out

    try:
        cls.convert_single_tensor_to_hf = convert_single_tensor_to_hf
        cls._maybe_stash_moe_split_spec = _maybe_stash_moe_split_spec
        cls._wrap_local_moe_stack = _wrap_local_moe_stack
        cls._aggregate_experts = _aggregate_experts
    except Exception as e:  # pragma: no cover
        logger.warning("automodel compat: failed to apply EP=1 expert-shard patch: %s", e)
        return False
    return True


# ---------------------------------------------------------------------------


def apply_automodel_compat_patches() -> None:
    """Apply all nemo_automodel runtime compatibility patches (idempotent)."""
    # Patch 4 runs FIRST and imports no nemo_automodel: the shim must be in
    # place before any vendored automodel import executes its transformers
    # imports.
    if "hub_resolve_revision" not in _applied:
        if _ensure_transformers_hub_resolve_revision():
            _applied.add("hub_resolve_revision")
            logger.info(
                "automodel compat: injected transformers.utils.hub.resolve_revision shim (image transformers < 5.18)"
            )

    key = "expert_slice_for_rank"
    if key not in _applied:
        if _patch_expert_slice_for_rank():
            _applied.add(key)
            logger.info("automodel compat: patched get_expert_slice_for_rank (PyTorch uneven-shard stride)")

    key = "dsv4_placeholder_global_shape"
    if key not in _applied:
        if _patch_dsv4_placeholder_global_shape():
            _applied.add(key)
            logger.info(
                "automodel compat: patched deepseek_v4 placeholder DTensor rebuilds "
                "(explicit global shape/stride for uneven FSDP2 shards)"
            )

    key = "dsv4_to_hf_iter"
    if key not in _applied:
        if _patch_dsv4_to_hf_iter():
            _applied.add(key)
            logger.info(
                "automodel compat: attached DeepSeekV4StateDictAdapter.to_hf_iter "
                "(streaming checkpoint-convention export for weight sync)"
            )

    key = "dequantize_from_fp8"
    if key not in _applied:
        if _patch_dequantize_from_fp8():
            _applied.add(key)
            logger.info(
                "automodel compat: patched dequantize_from_fp8 "
                "(true global shape re-wrap + 128-block-aligned padding for uneven shards)"
            )

    key = "dsv4_full_layer_ac"
    if key not in _applied:
        if _patch_dsv4_full_layer_activation_checkpointing():
            _applied.add(key)
            logger.info(
                "automodel compat: patched DeepseekV4ModelParallelizer "
                "(whole-layer activation checkpointing for the HC residual stream)"
            )

    key = "dsv4_sinkhorn_torch_fallback"
    if key not in _applied:
        if _patch_dsv4_sinkhorn_torch_fallback():
            _applied.add(key)
            logger.info(
                "automodel compat: patched dsv4_sinkhorn_normalize "
                "(graceful torch fallback when tile_kernels is missing)"
            )

    key = "dsv4_compress_ratios_compat"
    if key not in _applied:
        if _patch_dsv4_compress_ratios_compat():
            _applied.add(key)
            logger.info(
                "automodel compat: patched DeepseekV4ForCausalLM.from_config "
                "(rebuild compress_ratios from transformers>=5.x layer_types/compress_rates)"
            )

    # Runs last: its convert_single_tensor_to_hf wrapper chains on top of the
    # patch-2a wrapper applied above (both read the pristine upstream source
    # for their drift guards, so this ordering is required).
    key = "dsv4_plain_fsdp_expert_state_dict"
    if key not in _applied:
        if _patch_dsv4_plain_fsdp_expert_state_dict():
            _applied.add(key)
            logger.info(
                "automodel compat: patched DeepSeekV4StateDictAdapter "
                "(plain-FSDP2 EP=1 uneven expert-shard stash/wrap + contiguous .T export)"
            )
