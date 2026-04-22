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

from typing import Any, Optional

import logging
import os

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
        images_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        )
        mm["images_seqlens"] = images_seqlens
    return mm


def _compute_position_ids(
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: dict,
) -> torch.Tensor:
    """Re-compute position_ids for one trajectory (mirrors AgentLoopWorker)."""
    if processor is None or not multi_modal_inputs:
        return compute_position_id_with_mask(attention_mask)

    mm_kwargs: dict[str, Any] = {
        "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
        "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
    }
    if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
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
) -> DataProto:
    """Build one padded 1-row DataProto from a serialized intermediate trajectory.

    The tensor schema mirrors ``_InternalAgentLoopOutput``:
      - prompts, responses, response_mask, attention_mask, input_ids, position_ids
      - optional rollout_log_probs

    The non_tensor schema replicates (per-row) the fields of the parent
    RolloutSample's ``non_tensor_batch`` so that after ``DataProto.concat`` the
    resulting batch is schema-consistent with the main trajectory rows.
    """
    prompt_ids = traj["prompt_ids"]
    response_ids = traj["response_ids"]
    response_mask_raw = traj["response_mask"]
    response_logprobs_raw = traj.get("response_logprobs")
    routed_experts_raw = traj.get("routed_experts")
    multi_modal_data = traj.get("multi_modal_data")

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
        rollout_log_probs = torch.tensor(
            list(response_logprobs_raw) + [0.0] * pad_size, dtype=torch.float32
        ).unsqueeze(0)

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
            raise TypeError(
                f"Unsupported type for routed_experts: {type(routed_experts_raw)}"
            )
        length, layer_num, topk_num = experts_tensor.shape
        total_length = input_ids.shape[1]
        routed_experts_padded = torch.zeros(
            1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype
        )
        # Left-padded prompt: original prompt starts at (prompt_length - len(prompt_ids))
        start_pos = prompt_input_ids.shape[1] - len(prompt_ids)
        end_pos = min(start_pos + length, total_length)
        if start_pos < 0 or end_pos > total_length:
            raise ValueError(
                f"Invalid routed_experts position range: start_pos={start_pos}, "
                f"end_pos={end_pos}, total_length={total_length}"
            )
        routed_experts_padded[:, start_pos:end_pos] = experts_tensor[: end_pos - start_pos].unsqueeze(0)

    multi_modal_inputs = _compute_multi_modal_inputs(
        processor, tokenizer, multi_modal_data, input_ids
    )
    position_ids = _compute_position_ids(
        processor, input_ids, attention_mask, multi_modal_inputs
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
    if reward_score is not None:
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        # Place reward at last attended response position; fallback to last index
        last_idx = int(resp_attn[0].sum().item()) - 1
        last_idx = max(last_idx, 0)
        rm_scores[0, last_idx] = float(reward_score)
        tensor_batch["rm_scores"] = rm_scores

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
            "[IntermTrajExpander] no intermediate trajectories found in batch (size=%d), "
            "stripping empty column",
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
        "[IntermTrajExpander] Expanded %d main rows -> +%d intermediate rows "
        "(total %d); per-row counts=%s",
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
