# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Trainer-side expansion of ``intermediate_trajectories`` into DataProto rows.

This module implements the consumer end of the :class:`MultiTrajectoryAgentLoop`
packing protocol. It converts each serialized intermediate trajectory (stored
as a dict under ``non_tensor_batch["intermediate_trajectories"]``) into a 1-row
``DataProto`` whose tensor schema matches the main trajectory rows produced by
``AgentLoopWorker._agent_loop_postprocess``.

The padding logic mirrors the one in ``AgentLoopWorker._agent_loop_postprocess``
to keep the two ends consistent. It is deliberately kept as a standalone
utility (rather than reused from the worker) so that we do not need to modify
the ``AgentLoopWorker`` core interface.
"""

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.multi_trajectory_agent_loop import (
    INTERMEDIATE_TRAJECTORIES_KEY,
)
from verl.utils.model import compute_position_id_with_mask

# Data flow logger — imported lazily to avoid hard dependency.
try:
    from recipe.fully_async_gui_agent.data_flow_logger import log_dataproto, log_message

    _HAS_FLOW_LOG = True
except ImportError:
    _HAS_FLOW_LOG = False

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Batch schema assertion utility
# ---------------------------------------------------------------------------
def assert_batch_schema(
    data_proto: "DataProto",
    stage: str,
    *,
    expected_tensor_keys: set[str] | None = None,
    require_position_ids_ndim: int | None = None,
    has_processor: bool | None = None,
) -> None:
    """Assert structural invariants on a DataProto batch.

    Raises ``AssertionError`` with a precise diagnostic message on violation.
    Intended to be called at every major data-flow checkpoint so that shape /
    dtype / key-set mismatches surface immediately instead of propagating to
    a cryptic ``torch.cat`` or CUDA error many steps later.

    Args:
        data_proto: the batch to check.
        stage: human-readable label for the checkpoint (used in error messages).
        expected_tensor_keys: if given, assert these keys exist in ``batch.batch``.
        require_position_ids_ndim: if given, assert ``position_ids`` has this ndim
            on every row (catches the 2D-vs-3D mismatch for VL models).
        has_processor: if True, assert every intermediate trajectory carries
            non-empty ``multi_modal_data`` (VL model invariant).
    """
    n = len(data_proto)
    batch = data_proto.batch

    # 1. Basic non-empty check
    assert batch is not None, f"[{stage}] batch.batch is None (batch_size={n})"

    # 2. Batch-dim consistency: all tensors must have the same leading dim
    for k in batch.keys():
        t = batch[k]
        assert t.shape[0] == n, (
            f"[{stage}] tensor {k!r} has batch dim {t.shape[0]}, expected {n}"
        )

    # 3. Required keys
    if expected_tensor_keys:
        missing = expected_tensor_keys - set(batch.keys())
        assert not missing, f"[{stage}] missing tensor keys: {missing}"

    # 4. position_ids ndim consistency (critical for VL models)
    if require_position_ids_ndim is not None and "position_ids" in batch.keys():
        pos = batch["position_ids"]
        assert pos.ndim == require_position_ids_ndim, (
            f"[{stage}] position_ids.ndim={pos.ndim}, expected {require_position_ids_ndim}, "
            f"shape={tuple(pos.shape)}"
        )

    # 5. Seq-length consistency: input_ids, attention_mask, position_ids
    #    should share the same seq_len dimension.
    if "input_ids" in batch.keys() and "attention_mask" in batch.keys():
        seq_len = batch["input_ids"].shape[-1]
        am_len = batch["attention_mask"].shape[-1]
        assert seq_len == am_len, (
            f"[{stage}] input_ids seq_len={seq_len} != attention_mask seq_len={am_len}"
        )
        if "position_ids" in batch.keys():
            pos_seq = batch["position_ids"].shape[-1]
            assert pos_seq == seq_len, (
                f"[{stage}] position_ids seq_len={pos_seq} != input_ids seq_len={seq_len}, "
                f"position_ids.shape={tuple(batch['position_ids'].shape)}"
            )

    # 6. response_mask shape consistency
    if "response_mask" in batch.keys() and "responses" in batch.keys():
        rm_len = batch["response_mask"].shape[-1]
        resp_len = batch["responses"].shape[-1]
        assert rm_len == resp_len, (
            f"[{stage}] response_mask seq_len={rm_len} != responses seq_len={resp_len}"
        )

    # 7. non_tensor_batch length consistency
    nt = data_proto.non_tensor_batch or {}
    for k, v in nt.items():
        if hasattr(v, "__len__"):
            assert len(v) == n, (
                f"[{stage}] non_tensor_batch[{k!r}] length={len(v)}, expected {n}"
            )

    # 8. Intermediate trajectories cache validation
    meta = data_proto.meta_info or {}
    cache_key = "__intermediate_trajectories_cache__"
    cache = meta.get(cache_key)
    if cache is not None and isinstance(cache, dict):
        interm_col = cache.get("intermediate_col")
        main_bsz = cache.get("main_batch_size")

        # 8a. Cache main_batch_size must match actual batch size
        if main_bsz is not None:
            assert main_bsz == n, (
                f"[{stage}] intermediate cache main_batch_size={main_bsz} != batch_size={n}"
            )

        # 8b. intermediate_col length must match batch size
        if interm_col is not None:
            assert len(interm_col) == n, (
                f"[{stage}] intermediate_col length={len(interm_col)} != batch_size={n}"
            )

            # 8c. Each intermediate trajectory dict must have required keys
            #     and multi_modal_data when a VL processor is present
            for row_idx, row_list in enumerate(interm_col):
                if not row_list:
                    continue
                for traj_idx, traj in enumerate(row_list):
                    assert isinstance(traj, dict), (
                        f"[{stage}] intermediate_col[{row_idx}][{traj_idx}] "
                        f"is {type(traj).__name__}, expected dict"
                    )
                    for required_key in ("prompt_ids", "response_ids", "response_mask"):
                        assert required_key in traj, (
                            f"[{stage}] intermediate_col[{row_idx}][{traj_idx}] "
                            f"missing required key {required_key!r}"
                        )

                    # VL model: warn (but don't fail) if an intermediate
                    # trajectory has no images/videos. This can happen
                    # legitimately (e.g. text-only turn after a tool failure).
                    # _compute_position_ids handles this correctly.
                    if has_processor:
                        mm = traj.get("multi_modal_data")
                        _has_vision = bool(
                            mm and (mm.get("images") or mm.get("videos"))
                        )
                        if not _has_vision:
                            logger.warning(
                                "[%s] intermediate_col[%d][%d] has no "
                                "multi_modal_data (got %r). num_turns=%s",
                                stage, row_idx, traj_idx, mm,
                                traj.get("num_turns", "?"),
                            )


def _pad_to_length(
    tokenizer,
    ids: list[int],
    max_length: int,
    side: str,
    return_attention_mask: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pad a 1-D list of token ids to ``max_length`` and return ``(ids, mask)``.

    Mirrors the ``tokenizer.pad`` calls used by ``_agent_loop_postprocess``.
    """
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        out = tokenizer.pad(
            {"input_ids": ids},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=return_attention_mask,
        )
    finally:
        tokenizer.padding_side = prev_side

    input_ids = out["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = None
    if return_attention_mask:
        attention_mask = out["attention_mask"]
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask


def _compute_multi_modal_inputs(
    processor, tokenizer, multi_modal_data: Optional[dict], input_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Re-compute multi-modal inputs for one trajectory (mirrors AgentLoopWorker)."""
    if processor is None or not multi_modal_data:
        return {}

    images = multi_modal_data.get("images")
    videos = multi_modal_data.get("videos")

    video_metadatas = None
    if videos is not None:
        videos, video_metadatas = zip(*videos, strict=False)
        videos, video_metadatas = list(videos), list(video_metadatas)

    current_text = tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
    mm = processor(
        text=[current_text],
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_sample_frames=False,
    )
    mm.pop("input_ids", None)
    mm.pop("attention_mask", None)
    mm = dict(mm.convert_to_tensors("pt"))

    image_grid_thw = mm.get("image_grid_thw")
    if image_grid_thw is not None:
        images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
        mm["images_seqlens"] = images_seqlens
    return mm


def _compute_position_ids(
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: dict,
) -> torch.Tensor:
    """Re-compute position_ids for one trajectory (mirrors AgentLoopWorker).

    When ``processor`` is not None, always produces 3D position_ids
    ``(1, num_rope_axes+1, seq_len)`` — even when ``multi_modal_inputs`` is
    empty — so that the tensor schema is consistent with the main (final)
    trajectory rows produced by ``AgentLoopWorker._compute_position_ids``.
    This prevents ``torch.cat`` from failing with a "Tensors must have same
    number of dimensions" error during ``DataProto.concat``.
    """
    if processor is None:
        return compute_position_id_with_mask(attention_mask)

    mm_kwargs: dict[str, Any] = {
        "image_grid_thw": multi_modal_inputs.get("image_grid_thw") if multi_modal_inputs else None,
        "video_grid_thw": multi_modal_inputs.get("video_grid_thw") if multi_modal_inputs else None,
    }
    if multi_modal_inputs and multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[0][input_ids[0] == processor.image_token_id] = 1
        mm_token_type_ids[0][input_ids[0] == processor.video_token_id] = 2
        mm_kwargs["mm_token_type_ids"] = mm_token_type_ids

    vision_position_ids, _ = processor.get_rope_index(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **mm_kwargs,
    )
    vision_position_ids = vision_position_ids.transpose(0, 1)

    valid_mask = attention_mask[0].bool()
    text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
    text_position_ids = text_position_ids.unsqueeze(0)
    return torch.cat((text_position_ids, vision_position_ids), dim=1)


def _build_one_intermediate_row(
    traj: dict[str, Any],
    tokenizer,
    processor,
    rollout_config,
    inherited_non_tensor: dict[str, Any],
    inherited_tensors: Optional[dict[str, torch.Tensor]] = None,
    emit_rm_scores: bool = True,
) -> DataProto:
    """Build one padded 1-row DataProto from a serialized intermediate trajectory.

    The tensor schema mirrors ``_InternalAgentLoopOutput``:
      - prompts, responses, response_mask, attention_mask, input_ids, position_ids
      - optional rollout_log_probs

    The non_tensor schema replicates (per-row) the fields of the parent
    RolloutSample's ``non_tensor_batch`` so that after ``DataProto.concat`` the
    resulting batch is schema-consistent with the main trajectory rows.

    Args:
        inherited_tensors: optional dict of ``{field_name: tensor[1, L]}`` that
            the caller wants to inject verbatim onto the new row (e.g. inherit
            ``advantages`` / ``returns`` from the final trajectory of the same
            rollout). Shape is validated against ``response_length``.
        emit_rm_scores: when False, ``rm_scores`` is NOT written even if
            ``reward_score`` is present. Useful in the "post-advantage expand"
            path where reward has already been turned into advantage and must
            not be re-injected into ``token_level_scores``.
    """
    prompt_ids = traj["prompt_ids"]
    response_ids = traj["response_ids"]
    response_mask_raw = traj["response_mask"]
    response_logprobs_raw = traj.get("response_logprobs")
    routed_experts_raw = traj.get("routed_experts")
    multi_modal_data = traj.get("multi_modal_data")

    # Note: it is valid for an intermediate trajectory to have no
    # multi_modal_data (e.g. a text-only turn after a tool failure that
    # returned no screenshot). The _compute_position_ids function handles
    # this correctly by always producing 3D position_ids when a processor
    # is present, even without multi_modal_inputs. We log a warning so the
    # situation is visible but do NOT raise.
    if processor is not None:
        _has_images = bool(multi_modal_data and multi_modal_data.get("images"))
        _has_videos = bool(multi_modal_data and multi_modal_data.get("videos"))
        if not _has_images and not _has_videos:
            logger.warning(
                "[_build_one_intermediate_row] processor is present but intermediate "
                "trajectory has no multi_modal_data (got %r). "
                "prompt_ids_len=%d, response_ids_len=%d, num_turns=%s",
                multi_modal_data, len(prompt_ids), len(response_ids),
                traj.get("num_turns", "?"),
            )

    prompt_input_ids, prompt_attn = _pad_to_length(
        tokenizer, prompt_ids, rollout_config.prompt_length, "left", return_attention_mask=True
    )
    resp_input_ids, resp_attn = _pad_to_length(
        tokenizer, response_ids, rollout_config.response_length, "right", return_attention_mask=True
    )
    resp_mask_padded, _ = _pad_to_length(
        tokenizer,
        response_mask_raw,
        rollout_config.response_length,
        "right",
        return_attention_mask=False,
    )

    response_mask = resp_mask_padded * resp_attn
    attention_mask = torch.cat([prompt_attn, resp_attn], dim=1)
    input_ids = torch.cat([prompt_input_ids, resp_input_ids], dim=1)

    rollout_log_probs: Optional[torch.Tensor] = None
    if response_logprobs_raw is not None:
        pad_size = rollout_config.response_length - len(response_logprobs_raw)
        rollout_log_probs = torch.tensor(list(response_logprobs_raw) + [0.0] * pad_size, dtype=torch.float32).unsqueeze(
            0
        )

    # routed_experts: pad to full (prompt_length + response_length) like
    # AgentLoopWorker._agent_loop_postprocess does for the main trajectory.
    routed_experts_padded: Optional[torch.Tensor] = None
    if routed_experts_raw is not None:
        if isinstance(routed_experts_raw, np.ndarray):
            arr = routed_experts_raw
            if not arr.flags.writeable:
                arr = arr.copy()
            experts_tensor = torch.from_numpy(arr)
        elif isinstance(routed_experts_raw, torch.Tensor):
            experts_tensor = routed_experts_raw
        else:
            raise TypeError(f"Unsupported type for routed_experts: {type(routed_experts_raw)}")
        length, layer_num, topk_num = experts_tensor.shape
        total_length = input_ids.shape[1]
        routed_experts_padded = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)
        # Left-padded prompt: original prompt starts at (prompt_length - len(prompt_ids))
        start_pos = prompt_input_ids.shape[1] - len(prompt_ids)
        end_pos = min(start_pos + length, total_length)
        if start_pos < 0 or end_pos > total_length:
            raise ValueError(
                f"Invalid routed_experts position range: start_pos={start_pos}, "
                f"end_pos={end_pos}, total_length={total_length}"
            )
        routed_experts_padded[:, start_pos:end_pos] = experts_tensor[: end_pos - start_pos].unsqueeze(0)

    multi_modal_inputs = _compute_multi_modal_inputs(processor, tokenizer, multi_modal_data, input_ids)
    position_ids = _compute_position_ids(processor, input_ids, attention_mask, multi_modal_inputs)

    logger.info(
        "[_build_one_intermediate_row] multi_modal_data=%s, "
        "multi_modal_inputs_keys=%s, position_ids.shape=%s",
        "None" if multi_modal_data is None
        else ("empty" if not multi_modal_data else f"keys={list(multi_modal_data.keys())}"),
        list(multi_modal_inputs.keys()) if multi_modal_inputs else "empty",
        tuple(position_ids.shape),
    )

    # Build tensor batch
    tensor_batch: dict[str, torch.Tensor] = {
        "prompts": prompt_input_ids,
        "responses": resp_input_ids,
        "response_mask": response_mask,
        "attention_mask": attention_mask,
        "input_ids": input_ids,
        "position_ids": position_ids,
    }
    if rollout_log_probs is not None:
        tensor_batch["rollout_log_probs"] = rollout_log_probs
    if routed_experts_padded is not None:
        tensor_batch["routed_experts"] = routed_experts_padded

    # rm_scores: set the last response token's score to the shared reward.
    # Prefer the top-level ``reward_score`` field (AgentLoopOutput schema);
    # fall back to ``extra_fields.reward_score`` for backward compatibility.
    reward_score = traj.get("reward_score")
    if reward_score is None:
        reward_score = traj.get("extra_fields", {}).get("reward_score")
    if emit_rm_scores and reward_score is not None:
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        # Place reward at last attended response position; fallback to last index
        last_idx = int(resp_attn[0].sum().item()) - 1
        last_idx = max(last_idx, 0)
        rm_scores[0, last_idx] = float(reward_score)
        tensor_batch["rm_scores"] = rm_scores

    # Inject any tensor fields the caller wants to inherit verbatim from the
    # parent (final) row, e.g. ``advantages``, ``returns``,
    # ``token_level_rewards``, ``token_level_scores``. Each inherited tensor
    # must share the response_length dimension with ``response_mask``.
    if inherited_tensors:
        expected_resp_len = response_mask.shape[1]
        for k, v in inherited_tensors.items():
            if v is None:
                continue
            if v.dim() == 1:
                v = v.unsqueeze(0)
            if v.shape[-1] != expected_resp_len:
                raise ValueError(
                    f"Inherited tensor {k!r} has response length {v.shape[-1]}, expected {expected_resp_len}."
                )
            # Clone so that later in-place ops on the parent row do not mutate
            # the inherited copy (and vice versa).
            tensor_batch[k] = v.detach().clone()

    td = TensorDict(tensor_batch, batch_size=1)

    # Build non_tensor_batch:
    #   1) inherit per-row np-array fields from parent (one value per row).
    #   2) overlay the intermediate-trajectory-specific metadata.
    non_tensor_batch: dict[str, np.ndarray] = {}

    for k, v in inherited_non_tensor.items():
        non_tensor_batch[k] = v

    # Overlay only the keys that the main row also carries in its
    # ``non_tensor_batch`` (populated by ``_combine_agent_loop_outputs``).
    # Adding extra keys here (e.g. from ``traj_extra``) would cause
    # ``DataProto.concat`` to fail because the main row would be missing
    # those keys.  Fields like ``reward_score`` are already expressed via
    # the ``rm_scores`` tensor and do not need a non_tensor duplicate.
    traj_extra = traj.get("extra_fields", {}) or {}
    # Only forward extra_fields keys that the parent (main) row already has.
    parent_keys = set(inherited_non_tensor.keys())
    overlay = {
        "trajectory_role": "intermediate",
        "turn_number": int(traj.get("num_turns", 0)),
    }
    for k, v in traj_extra.items():
        if k in parent_keys or k in ("trajectory_role", "turn_number"):
            overlay[k] = v

    for k, v in overlay.items():
        arr = np.empty(1, dtype=object)
        arr[0] = v
        non_tensor_batch[k] = arr

    # Multi-modal inputs attached as object array, matching AgentLoopWorker convention.
    # Always emit this key (with an empty dict if needed) so that the schema
    # stays consistent with the main row produced by ``_combine_agent_loop_outputs``.
    mm_arr = np.empty(1, dtype=object)
    mm_arr[0] = multi_modal_inputs if multi_modal_inputs else {}
    non_tensor_batch["multi_modal_inputs"] = mm_arr

    return DataProto(batch=td, non_tensor_batch=non_tensor_batch)


def _slice_parent_non_tensor(parent_nt: dict[str, np.ndarray], row_idx: int) -> dict[str, np.ndarray]:
    """Return a per-row slice of parent non_tensor_batch as single-row np arrays.

    Fields whose length does not equal the parent batch size are excluded
    (e.g., shared scalar metadata); the caller will re-broadcast them if
    needed. ``intermediate_trajectories`` is never forwarded to avoid
    recursive expansion.
    """
    result: dict[str, np.ndarray] = {}
    batch_size = None
    # Infer batch size from the parent's tensor batch if available
    if parent_nt:
        batch_size = next((len(v) for v in parent_nt.values() if hasattr(v, "__len__")), None)

    for k, v in parent_nt.items():
        if k == INTERMEDIATE_TRAJECTORIES_KEY:
            continue
        try:
            if batch_size is not None and len(v) == batch_size:
                arr = np.empty(1, dtype=object)
                arr[0] = v[row_idx]
                result[k] = arr
            else:
                # Keep as-is if it's a scalar-ish field
                arr = np.empty(1, dtype=object)
                arr[0] = v
                result[k] = arr
        except TypeError:
            arr = np.empty(1, dtype=object)
            arr[0] = v
            result[k] = arr
    return result


def expand_intermediate_trajectories(
    data_proto: DataProto,
    tokenizer,
    processor,
    rollout_config,
) -> DataProto:
    """Expand ``intermediate_trajectories`` into extra rows of ``data_proto``.

    ``data_proto`` is the ``RolloutSample.full_batch`` after ``addition_process``.
    It has batch size ``N = rollout.n`` (one prompt × n samples). Each of the N
    rows may carry its own ``intermediate_trajectories`` list under
    ``non_tensor_batch``.

    For every intermediate trajectory on row i, a new padded 1-row DataProto
    is built (inheriting row-i's non-tensor context) and appended to the
    result. The result is then ``DataProto.concat`` of:

        [data_proto, row0_interm0, row0_interm1, ..., row1_interm0, ...]

    If no row carries any intermediate trajectory, ``data_proto`` is returned
    unchanged (transparent to non-GUI-Agent callers).
    """
    nt = data_proto.non_tensor_batch or {}
    if INTERMEDIATE_TRAJECTORIES_KEY not in nt:
        return data_proto

    interm_col = nt[INTERMEDIATE_TRAJECTORIES_KEY]
    # Handle both np.ndarray(object) and plain list form.
    if hasattr(interm_col, "__len__"):
        rows_with_interm = any(bool(x) for x in interm_col)
    else:
        rows_with_interm = False
    if not rows_with_interm:
        # Strip the empty column and return as-is.
        logger.debug(
            "[IntermTrajExpander] no intermediate trajectories found in batch (size=%d), stripping empty column",
            len(data_proto),
        )
        stripped = DataProto(
            batch=data_proto.batch,
            non_tensor_batch={k: v for k, v in nt.items() if k != INTERMEDIATE_TRAJECTORIES_KEY},
            meta_info=dict(data_proto.meta_info),
        )
        return stripped

    n_rows = len(interm_col)
    pieces: list[DataProto] = [data_proto]
    total_appended = 0
    per_row_counts: list[int] = []

    for row_idx in range(n_rows):
        interm_list = interm_col[row_idx]
        if not interm_list:
            per_row_counts.append(0)
            continue
        inherited_nt = _slice_parent_non_tensor(nt, row_idx)
        for traj in interm_list:
            piece = _build_one_intermediate_row(
                traj=traj,
                tokenizer=tokenizer,
                processor=processor,
                rollout_config=rollout_config,
                inherited_non_tensor=inherited_nt,
            )
            pieces.append(piece)
            total_appended += 1
        per_row_counts.append(len(interm_list))

    # Strip intermediate_trajectories from main data_proto so the concat is clean.
    main = DataProto(
        batch=data_proto.batch,
        non_tensor_batch={k: v for k, v in nt.items() if k != INTERMEDIATE_TRAJECTORIES_KEY},
        meta_info=dict(data_proto.meta_info),
    )
    pieces[0] = main

    expanded = DataProto.concat(pieces)
    logger.info(
        "[IntermTrajExpander] Expanded %d main rows -> +%d intermediate rows (total %d); per-row counts=%s",
        n_rows,
        total_appended,
        len(expanded),
        per_row_counts,
    )

    if _HAS_FLOW_LOG:
        log_dataproto(
            expanded,
            stage="expand_intermediate_trajectories.output",
            extra={
                "n_main_rows": n_rows,
                "n_intermediate_appended": total_appended,
                "per_row_counts": per_row_counts,
            },
        )

    return expanded


def strip_intermediate_trajectories_column(
    data_proto: DataProto,
    *,
    cache_key: str = "__intermediate_trajectories_cache__",
) -> DataProto:
    """Move ``intermediate_trajectories`` from ``non_tensor_batch`` into ``meta_info``.

    This is used at the *assemble* stage in the post-advantage-expand pipeline:
    we keep the serialized intermediate payload alive (to be expanded later,
    after advantage is computed), but we remove it from ``non_tensor_batch`` so
    that the standard downstream stages (balance_batch, advantage, actor
    update) see a clean batch of size ``N = rollout.n * num_rollouts``.

    The payload is cached in ``meta_info[cache_key]`` as a tuple of
    ``(intermediate_col, main_batch_size)`` so that the downstream expander
    can find it.
    """
    nt = data_proto.non_tensor_batch or {}
    if INTERMEDIATE_TRAJECTORIES_KEY not in nt:
        return data_proto

    interm_col = nt[INTERMEDIATE_TRAJECTORIES_KEY]
    meta_info = dict(data_proto.meta_info)
    meta_info[cache_key] = {
        "intermediate_col": interm_col,
        "main_batch_size": len(data_proto),
    }
    stripped = DataProto(
        batch=data_proto.batch,
        non_tensor_batch={k: v for k, v in nt.items() if k != INTERMEDIATE_TRAJECTORIES_KEY},
        meta_info=meta_info,
    )
    return stripped


def _compute_rollout_group_ids(
    data_proto: DataProto,
    rollout_n: int,
) -> np.ndarray:
    """Assign each row to a rollout-group id.

    Semantics: "rollout group" means one prompt (one RolloutSample) that was
    repeated ``rollout.n`` times at generation time. We identify it by the
    ``uid`` non-tensor field set in ``FullyAsyncRollouter`` (same uid for all
    ``rollout.n`` rows of the same RolloutSample). If ``uid`` is absent, fall
    back to bucketing by a fixed stride of ``rollout_n``.

    Returns a numpy array of shape (batch_size,) with dense integer group ids.
    """
    bsz = len(data_proto)
    nt = data_proto.non_tensor_batch or {}
    if "uid" in nt and len(nt["uid"]) == bsz:
        uids = nt["uid"]
        # map unique uid -> dense id, preserving first-seen order
        order: dict = {}
        group_ids = np.empty(bsz, dtype=np.int64)
        for i, u in enumerate(uids):
            if u not in order:
                order[u] = len(order)
            group_ids[i] = order[u]
        return group_ids
    # Fallback: pure positional bucketing. Assumes rows are interleaved in
    # contiguous groups of size rollout_n, which matches the way
    # ``prepare_single_generation_data`` repeats each prompt.
    if rollout_n <= 0:
        rollout_n = 1
    return np.arange(bsz, dtype=np.int64) // rollout_n


def expand_intermediate_trajectories_pre_log_prob(
    data_proto: DataProto,
    tokenizer,
    processor,
    rollout_config,
    *,
    rollout_n: int = 1,
    cache_key: str = "__intermediate_trajectories_cache__",
) -> DataProto:
    """Expand intermediate trajectories BEFORE log_prob / ref / critic forward.

    Placement: run right after reward extraction, before any per-token forward
    pass. The expanded batch then feeds log_prob / ref_log_prob / critic so
    that every row (final + intermediate) receives its own correct per-token
    tensors from independent forward passes.

    What it does
    ------------
    * For every intermediate trajectory stored on a final row, build a new
      padded 1-row DataProto that matches the final row's tensor schema at
      this stage (``prompts``, ``responses``, ``response_mask``,
      ``attention_mask``, ``input_ids``, ``position_ids``, optional
      ``rollout_log_probs`` / ``routed_experts``, and ``rm_scores``-free
      since reward is already aggregated on the final row only).
    * Stamp two non-tensor fields on every row (final *and* intermediate):
        - ``trajectory_role`` = ``"final"`` | ``"intermediate"``
        - ``rollout_group_id`` = dense integer id identifying the rollout
          (same for all rows that belong to the same RolloutSample).
      These labels are used later by :func:`_fit_compute_advantage` to
      (a) pick the final-only subset for GRPO statistics, and (b) compute
      ``1 / T_rollout`` normalization after scattering advantage.

    No ``advantages`` / ``returns`` inheritance happens here (advantage is
    still to be computed). No ``rm_scores`` is emitted on intermediate rows
    either — the final row already carries the shared reward, and advantage
    will be computed on final-only, so intermediate rows do not need any
    token-level reward of their own.

    The input payload is expected to live in ``meta_info[cache_key]`` (put
    there by :func:`strip_intermediate_trajectories_column`). If the cache is
    missing, the batch is returned unchanged (with ``trajectory_role`` /
    ``rollout_group_id`` stamped on final rows so the downstream code can
    still rely on them).
    """
    meta_info = dict(data_proto.meta_info)
    cache = meta_info.pop(cache_key, None)
    nt = dict(data_proto.non_tensor_batch or {})
    # Normalize carrier: legacy path may still stash the payload in
    # non_tensor_batch directly.
    interm_col = None
    if cache is not None:
        interm_col = cache.get("intermediate_col")
        cached_main_bsz = cache.get("main_batch_size", len(data_proto))
        if cached_main_bsz != len(data_proto):
            logger.warning(
                "[IntermTrajExpander.pre_logprob] cached main_batch_size=%d differs "
                "from current batch size=%d; assuming row order is preserved",
                cached_main_bsz,
                len(data_proto),
            )
    elif INTERMEDIATE_TRAJECTORIES_KEY in nt:
        interm_col = nt.pop(INTERMEDIATE_TRAJECTORIES_KEY)

    base = DataProto(
        batch=data_proto.batch,
        non_tensor_batch=nt,
        meta_info=meta_info,
    )
    n_rows = len(base)

    # --- Assert: validate base batch schema before expansion ---
    _pos_ndim = base.batch["position_ids"].ndim if "position_ids" in base.batch.keys() else None
    assert_batch_schema(base, "expand_pre_logprob.base_before_expand", require_position_ids_ndim=_pos_ndim)

    # Group ids for the final rows (uid-based; falls back to positional buckets).
    group_ids = _compute_rollout_group_ids(base, rollout_n=rollout_n)

    # Stamp role / group id on the final rows even when no intermediates are
    # present, so downstream code can always rely on these fields.
    def _stamp_role_and_group(dp: DataProto, roles: np.ndarray, gids: np.ndarray) -> None:
        dp.non_tensor_batch["trajectory_role"] = roles
        dp.non_tensor_batch["rollout_group_id"] = gids.astype(np.int64, copy=False)

    if interm_col is None or len(interm_col) != n_rows or not any(bool(x) for x in interm_col):
        if interm_col is not None and len(interm_col) != n_rows:
            logger.warning(
                "[IntermTrajExpander.pre_logprob] intermediate column length (%d) "
                "does not match batch size (%d); skipping expansion",
                len(interm_col),
                n_rows,
            )
        roles = np.array(["final"] * n_rows, dtype=object)
        _stamp_role_and_group(base, roles, group_ids)
        base.meta_info["fully_async/intermediate/num_final_rows"] = int(n_rows)
        base.meta_info["fully_async/intermediate/num_intermediate_rows"] = 0
        return base

    # Build intermediate pieces.
    per_row_counts: list[int] = []
    pieces: list[DataProto] = [base]
    for row_idx in range(n_rows):
        raw_list = interm_col[row_idx]
        if not raw_list:
            per_row_counts.append(0)
            continue
        inherited_nt = _slice_parent_non_tensor(nt, row_idx)
        for traj in raw_list:
            piece = _build_one_intermediate_row(
                traj=traj,
                tokenizer=tokenizer,
                processor=processor,
                rollout_config=rollout_config,
                inherited_non_tensor=inherited_nt,
                inherited_tensors=None,
                emit_rm_scores=False,
            )
            # --- Assert: each intermediate piece must match base position_ids ndim ---
            if _pos_ndim is not None and "position_ids" in piece.batch.keys():
                piece_pos_ndim = piece.batch["position_ids"].ndim
                assert piece_pos_ndim == _pos_ndim, (
                    f"[expand_pre_logprob] intermediate piece position_ids.ndim={piece_pos_ndim} "
                    f"!= base ndim={_pos_ndim}, piece position_ids.shape="
                    f"{tuple(piece.batch['position_ids'].shape)}, row_idx={row_idx}"
                )
            pieces.append(piece)
        per_row_counts.append(len(raw_list))

    total_appended = sum(per_row_counts)
    if total_appended == 0:
        roles = np.array(["final"] * n_rows, dtype=object)
        _stamp_role_and_group(base, roles, group_ids)
        return base

    # Stamp role/group on each piece individually BEFORE concat. This keeps
    # the concat schema consistent across all rows.
    final_roles = np.array(["final"] * n_rows, dtype=object)
    _stamp_role_and_group(base, final_roles, group_ids)

    # Iterate pieces[1:] in the order they were appended, mapping each back
    # to its parent row.
    piece_idx = 1
    for row_idx in range(n_rows):
        cnt = per_row_counts[row_idx]
        for _ in range(cnt):
            p = pieces[piece_idx]
            roles_p = np.array(["intermediate"], dtype=object)
            gids_p = np.array([group_ids[row_idx]], dtype=np.int64)
            _stamp_role_and_group(p, roles_p, gids_p)
            piece_idx += 1

    # --------------------------------------------------------------
    # Align tensor schema across pieces BEFORE concat.
    #
    # Intermediate rows are built from a minimal tensor schema, but the
    # (final) base batch may have already accumulated extra per-token tensors
    # from earlier stages of fit_step (e.g. ``rm_scores`` written by
    # ``_fit_compute_reward``, or leftover bookkeeping fields). ``torch.cat``
    # via tensordict requires identical key sets, so we fill any missing
    # tensor on intermediate pieces with a zero-valued tensor of matching
    # shape/dtype. These zero fills are never treated as training signal:
    # advantage is computed on the FINAL subset only, and padding /
    # response_mask guards any downstream aggregation.
    # --------------------------------------------------------------
    base_keys = set(base.batch.keys())
    for p in pieces[1:]:
        piece_keys = set(p.batch.keys())
        # (a) fields present in base but missing on intermediate piece: fill 0
        for k in base_keys - piece_keys:
            ref = base.batch[k]
            # shape: keep everything except batch dim, which is 1 for a piece
            shape = (1,) + tuple(ref.shape[1:])
            p.batch[k] = torch.zeros(shape, dtype=ref.dtype)
        # (b) fields present on piece but missing in base: fill 0 on base.
        # This should be rare, but keeps the contract symmetric.
        extra = piece_keys - base_keys
        if extra:
            for k in extra:
                ref = p.batch[k]
                shape = (len(base),) + tuple(ref.shape[1:])
                base.batch[k] = torch.zeros(shape, dtype=ref.dtype)
            base_keys = set(base.batch.keys())

    # Same alignment for non_tensor_batch, since DataProto.concat runs a
    # plain np.concatenate over each key and will blow up if any key is
    # missing on any piece.
    base_nt_keys = set(base.non_tensor_batch.keys())
    for p in pieces[1:]:
        piece_nt_keys = set(p.non_tensor_batch.keys())
        for k in base_nt_keys - piece_nt_keys:
            p.non_tensor_batch[k] = np.array([None], dtype=object)
        extra_nt = piece_nt_keys - base_nt_keys
        if extra_nt:
            for k in extra_nt:
                base.non_tensor_batch[k] = np.array([None] * len(base), dtype=object)
            base_nt_keys = set(base.non_tensor_batch.keys())

    # Pre-concat sanity check: every shared tensor key must have matching
    # ndim / trailing shape / dtype across all pieces. Fail immediately with
    # a precise diagnostic instead of letting ``torch.cat`` raise a generic
    # "Tensors must have same number of dimensions".
    for k in base.batch.keys():
        ref = base.batch[k]
        for pi, p in enumerate(pieces[1:], start=1):
            if k not in p.batch:
                continue
            got = p.batch[k]
            assert got.dim() == ref.dim() and tuple(got.shape[1:]) == tuple(ref.shape[1:]) and got.dtype == ref.dtype, (
                f"[expand_pre_logprob] tensor schema mismatch for key={k!r} "
                f"between base and piece[{pi}]: "
                f"base.shape={tuple(ref.shape)} base.dtype={ref.dtype} vs "
                f"piece.shape={tuple(got.shape)} piece.dtype={got.dtype}"
            )

    expanded = DataProto.concat(pieces)

    # --- Assert: validate expanded batch schema ---
    assert_batch_schema(
        expanded, "expand_pre_logprob.after_concat",
        require_position_ids_ndim=_pos_ndim,
    )

    expanded.meta_info["fully_async/intermediate/num_final_rows"] = int(n_rows)
    expanded.meta_info["fully_async/intermediate/num_intermediate_rows"] = int(total_appended)

    logger.info(
        "[IntermTrajExpander.pre_logprob] Expanded %d final rows -> +%d intermediate "
        "rows (total %d); per-row counts=%s",
        n_rows,
        total_appended,
        len(expanded),
        per_row_counts,
    )

    if _HAS_FLOW_LOG:
        log_dataproto(
            expanded,
            stage="expand_intermediate_trajectories_pre_log_prob.output",
            extra={
                "n_final_rows": n_rows,
                "n_intermediate_appended": total_appended,
                "per_row_counts": per_row_counts,
            },
        )

    return expanded


def zero_out_padding_rows(data_proto: DataProto, pad_size: int) -> DataProto:
    """Zero out the training signal on the last ``pad_size`` rows.

    ``pad_dataproto_to_divisor`` pads a DataProto by repeating rows (head
    slice), which is fine to satisfy shape constraints and pass a forward
    pass, but those rows must not contribute to the actor loss AND must not
    be mistaken for real final/intermediate rows during advantage compute.

    We therefore:
      * zero ``response_mask`` (so token-mean ignores them),
      * zero ``advantages`` / ``returns`` / ``rm_scores`` /
        ``token_level_{rewards,scores}`` (belt-and-braces), and
      * relabel ``trajectory_role`` -> ``"padding"`` and
        ``rollout_group_id`` -> ``-1`` on the padded tail, so downstream
        role-aware logic (``_fit_compute_advantage`` slicing by
        ``role == "final"``, ``scatter_advantage_to_intermediate_and_normalize``
        grouping by ``rollout_group_id``) skips them cleanly.
    """
    if pad_size <= 0:
        return data_proto

    total = len(data_proto)
    start = total - pad_size
    fields_to_zero = (
        "response_mask",
        "advantages",
        "returns",
        "rm_scores",
        "token_level_rewards",
        "token_level_scores",
    )
    for k in fields_to_zero:
        if k in data_proto.batch.keys():
            data_proto.batch[k][start:total] = 0

    # Relabel role/group on the padded tail so role-aware stages do not
    # misinterpret the repeated head rows.
    nt = data_proto.non_tensor_batch or {}
    if "trajectory_role" in nt:
        roles = np.asarray(nt["trajectory_role"], dtype=object).copy()
        roles[start:total] = "padding"
        nt["trajectory_role"] = roles
    if "rollout_group_id" in nt:
        gids = np.asarray(nt["rollout_group_id"], dtype=np.int64).copy()
        gids[start:total] = -1
        nt["rollout_group_id"] = gids

    return data_proto


def scatter_advantage_to_intermediate_and_normalize(
    data_proto: DataProto,
    *,
    normalize_rollout_weight: bool = True,
) -> DataProto:
    """Scatter final-row advantage/returns to sibling intermediate rows and
    apply ``1 / T_rollout`` normalization.

    Assumes the expanded batch has two non-tensor fields stamped by
    :func:`expand_intermediate_trajectories_pre_log_prob`:
      - ``trajectory_role`` ∈ {"final", "intermediate"}
      - ``rollout_group_id`` (int)

    And that ``advantages`` / ``returns`` are already set on *final* rows
    (computed by the standard GRPO path on the final subset). This function:

    1. Copies the per-group final-row advantages/returns onto the intermediate
       rows sharing the same ``rollout_group_id``. Each rollout group has
       ``rollout.n`` final rows (all carry the same advantage scalar in GRPO
       after group-normalize), so we pick the first final row in each group
       as the canonical source.
    2. Optionally multiplies every row in a rollout group by ``1 / T_rollout``
       where ``T_rollout`` is the total number of valid response tokens across
       that rollout's final + all intermediate rows. Under
       ``loss_agg_mode="token-mean"`` this yields equal per-rollout
       contribution to the final loss regardless of how many turns it has.
    """
    if "advantages" not in data_proto.batch.keys():
        return data_proto
    nt = data_proto.non_tensor_batch or {}
    if "trajectory_role" not in nt or "rollout_group_id" not in nt:
        # Expander wasn't run (or no intermediate was present); treat the
        # batch as final-only and still apply rollout normalization if
        # requested. In that case each rollout has rollout.n rows and
        # T_rollout is just the sum of their response-token counts.
        if not normalize_rollout_weight:
            return data_proto

    n = len(data_proto)
    roles = nt.get("trajectory_role")
    group_ids = nt.get("rollout_group_id")
    if group_ids is None:
        return data_proto
    group_ids_np = np.asarray(group_ids, dtype=np.int64)

    # ------------------------------------------------------------------
    # (1) Scatter advantages / returns from final rows to intermediate rows.
    # ------------------------------------------------------------------
    if roles is not None:
        # Build: group_id -> first-final-row-index
        group_to_final: dict[int, int] = {}
        for i in range(n):
            if roles[i] == "final":
                gid = int(group_ids_np[i])
                if gid not in group_to_final:
                    group_to_final[gid] = i

        # Copy advantage vectors from the canonical final row onto
        # intermediate rows of the same group. We do this as whole-row
        # copies to preserve any in-row structure (though for GRPO the
        # scalar is broadcast across response_length on all rows anyway).
        adv = data_proto.batch["advantages"]
        has_returns = "returns" in data_proto.batch.keys()
        ret = data_proto.batch["returns"] if has_returns else None

        for i in range(n):
            if roles[i] != "intermediate":
                continue
            gid = int(group_ids_np[i])
            src = group_to_final.get(gid)
            if src is None:
                # Orphan intermediate (no final row in the same group).
                # Zero out the advantage to be safe.
                adv[i] = 0
                if has_returns:
                    ret[i] = 0
                continue
            adv[i] = adv[src]
            if has_returns:
                ret[i] = ret[src]

    # ------------------------------------------------------------------
    # (2) Rollout-level weight normalization: advantage /= T_rollout.
    # ------------------------------------------------------------------
    if normalize_rollout_weight:
        if "response_mask" not in data_proto.batch.keys():
            logger.warning("[scatter_advantage] response_mask missing; skipping 1/T_rollout normalization")
        else:
            response_mask = data_proto.batch["response_mask"]
            per_row_tokens = response_mask.to(torch.float32).sum(dim=-1)
            per_row_tokens_np = per_row_tokens.detach().cpu().numpy()

            # Aggregate token count per group.
            group_tokens: dict[int, float] = {}
            for i in range(n):
                gid = int(group_ids_np[i])
                group_tokens[gid] = group_tokens.get(gid, 0.0) + float(per_row_tokens_np[i])

            # Build scale[i] = 1 / T_rollout_of(i).
            adv_dtype = data_proto.batch["advantages"].dtype
            scale = torch.empty(n, dtype=adv_dtype)
            for i in range(n):
                gid = int(group_ids_np[i])
                t = group_tokens.get(gid, 0.0)
                scale[i] = (1.0 / t) if t > 0 else 0.0
            scale = scale.unsqueeze(-1)

            data_proto.batch["advantages"] = data_proto.batch["advantages"] * scale
            if "returns" in data_proto.batch.keys():
                data_proto.batch["returns"] = data_proto.batch["returns"] * scale

            data_proto.meta_info["fully_async/rollout_weight/num_groups"] = len(group_tokens)
            data_proto.meta_info["fully_async/rollout_weight/total_tokens"] = float(sum(group_tokens.values()))

    return data_proto
