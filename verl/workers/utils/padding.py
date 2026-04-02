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

"""Padding / nesting utilities for verl worker pipelines.

Two APIs coexist here:

* **Legacy** — :func:`left_right_2_no_padding` and
  :func:`no_padding_2_padding`. Depend on ``flash_attn.unpad_input`` and
  only nest ``input_ids`` / ``position_ids``. Still used when
  ``RayPPOTrainer.use_mask_nesting`` is ``False`` (the current default).
  Pending removal once the new API becomes the default and stabilises.

* **Current** — :func:`nest_batch_by_mask` / :func:`unnest_batch_by_mask`
  for the per-batch nest/unnest roundtrip, :func:`extract_response`
  to slice response regions from model outputs (dispatches over both
  styles). Backed by :class:`verl.utils.nested_tensor.MaskNestingSpec`.
  Does not depend on ``flash_attn``.
"""

import enum
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import index_first_axis, unpad_input
from verl.utils.nested_tensor import (
    MaskNestingSpec,
    _rle_scatter_indices,
    rle_to_mask,
)


def left_right_2_no_padding(data: TensorDict) -> TensorDict:
    """
    Convert TensorDict from left-right padding to no-padding format.

    .. note::
        Legacy API kept for the ``use_mask_nesting=False`` code path.
        New code should prefer :func:`nest_batch`.

    Args:
        data: TensorDict with "input_ids", "attention_mask", "response_mask", "position_ids"

    Returns:
        data: TensorDict with
        - Tensor includes NestedTensors like "input_ids", "loss_mask", "position_ids"
        - NonTensorData includes "max_seq_len", "max_response_len", "indices"

    Note:
    1. the return input_ids/position_ids/loss_mask are nested tensor.
    2. we will remove "attention_mask", "response" in the return data, but "response_mask" is kept.
    """
    assert "input_ids" in data, "input_ids is required in left-right padding data"
    assert "attention_mask" in data, "attention_mask is required in left-right padding data"
    assert "response_mask" in data, "response_mask is required in left-right padding data"
    assert "position_ids" in data, "position_ids is required in left-right padding data"

    input_ids = data.pop("input_ids")
    attention_mask = data["attention_mask"]
    response_mask = data["response_mask"]
    position_ids = data["position_ids"]  # (bs, seq_len) or # (bs, 4, seq_len)

    max_seq_len, max_response_len = input_ids.shape[1], response_mask.shape[1]
    tu.assign_non_tensor_data(data, "max_seq_len", max_seq_len)
    tu.assign_non_tensor_data(data, "max_response_len", max_response_len)

    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    tu.assign_non_tensor_data(data, "indices", indices)

    input_ids_nested = torch.nested.nested_tensor_from_jagged(input_ids_rmpad.squeeze(-1), offsets=cu_seqlens)

    position_ids_list = []
    for i in range(attention_mask.shape[0]):
        curr_mask = attention_mask[i].bool()
        curr_pos_ids = position_ids[i]
        if curr_pos_ids.dim() == 1:  # (seq_len,)
            valid_ids = curr_pos_ids[curr_mask]
        else:  # (4, seq_len)
            valid_ids = curr_pos_ids[:, curr_mask]
        position_ids_list.append(valid_ids)
    position_ids_nested = torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)

    data["input_ids"] = input_ids_nested
    data["position_ids"] = position_ids_nested
    data["loss_mask"] = data["response_mask"]

    routed_experts = data.get("routed_experts", None)
    if routed_experts is not None and not routed_experts.is_nested:
        # uint8 range is [0, 255]; guard against negative sentinels that would
        # wrap-around on cast.
        if routed_experts.min() >= 0 and routed_experts.max() <= 255:
            routed_experts = routed_experts.to(torch.uint8)
        routed_experts_rmpad = index_first_axis(routed_experts.unsqueeze(-1).flatten(0, 1), indices)
        routed_experts_nested = torch.nested.nested_tensor_from_jagged(
            routed_experts_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        data["routed_experts"] = routed_experts_nested

    # (bsz, seqlen, topk)
    teacher_logprobs = data.get("teacher_logprobs", None)
    teacher_ids = data.get("teacher_ids", None)
    if teacher_logprobs is not None and teacher_ids is not None:
        teacher_logprobs_rmpad = index_first_axis(teacher_logprobs.unsqueeze(-1).flatten(0, 1), indices)
        teacher_ids_rmpad = index_first_axis(teacher_ids.unsqueeze(-1).flatten(0, 1), indices)
        teacher_logprobs_nested = torch.nested.nested_tensor_from_jagged(
            teacher_logprobs_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        teacher_ids_nested = torch.nested.nested_tensor_from_jagged(teacher_ids_rmpad.squeeze(-1), offsets=cu_seqlens)
        data["teacher_logprobs"] = teacher_logprobs_nested
        data["teacher_ids"] = teacher_ids_nested

    return data


def no_padding_2_padding(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Slice response from unpad model output.

    .. note::
        Legacy API kept for the ``use_mask_nesting=False`` code path.
        New code should prefer :func:`unnest` + :func:`slice_response`.

    Args:
        tensor: a nested tensor or a tensor of shape (total_nnz,*),
            total_nnz is the total number of tokens across all sequences in the batch

        data: TensorDict with "prompts", "responses", "attention_mask"

    Returns:
        tensor: sliced response tensor of shape [bsz, max_response_len, *]
    """
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_response_len < 0:
            max_response_len = response_lens.max().item()
    else:
        attention_mask = data["attention_mask"]
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]
    assert not prompt_lens.eq(0).any(), f"seq_offset - resp_len - 1 assumes prompt_len > 0. Got {prompt_lens}"

    response_list = []
    # Skip padding dimensions after sequence dimensions, if any.
    skip_padding = (0, 0) * (values.ndim - 1)
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        # left-shift model output by one token for log_probs/values
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (*skip_padding, 0, pad_size)))

    output = torch.stack(response_list, dim=0)
    return output


def embeds_padding_2_no_padding(data: TensorDict) -> TensorDict:
    """
    Convert TensorDict from prompt embeds with padding to no-padding format.
    For diffusion model training only.

    Currently we expect the prompt embedding mask to be [1111000...] format,
    which means the valid tokens are continuous and start from the left.

    Args:
        data: TensorDict with "prompt_embeds", "prompt_embeds_mask",
              "negative_prompt_embeds", "negative_prompt_embeds_mask"

    Returns:
        data: TensorDict with
        - Tensor includes NestedTensors "prompt_embeds", "prompt_embeds_mask",
          "negative_prompt_embeds", "negative_prompt_embeds_mask"
    """

    def _to_nested(embeds: torch.Tensor, mask: torch.Tensor):
        """Strip padding from (bs, seq_len, dim) embeds using the boolean mask and return nested tensors."""
        embeds_list, mask_list = [], []
        for i in range(mask.shape[0]):
            curr_mask = mask[i].bool()
            embeds_list.append(embeds[i, curr_mask, :])
            mask_list.append(curr_mask[curr_mask])
        return (
            torch.nested.as_nested_tensor(embeds_list, layout=torch.jagged),
            torch.nested.as_nested_tensor(mask_list, layout=torch.jagged),
        )

    data["prompt_embeds"], data["prompt_embeds_mask"] = _to_nested(data["prompt_embeds"], data["prompt_embeds_mask"])

    if isinstance(data.get("negative_prompt_embeds", None), torch.Tensor):
        data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"] = _to_nested(
            data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"]
        )

    return data


class DynamicPadValue(enum.Enum):
    """Sentinel enum for pad values that must be resolved at nesting time.

    Registry entries tagged with a member of this enum defer their pad
    value to a caller-supplied kwarg of :func:`nest_batch_by_mask`. Each
    member corresponds to a specific tokenizer-owned value so the
    library can declare *what kind* of dynamic value a field needs
    without hard-coding the value itself.

    verl currently only uses :attr:`PAD_TOKEN_ID`. The enum leaves room
    for additional dynamic kinds (``BOS_TOKEN_ID``, ``EOS_TOKEN_ID``,
    ``MASK_TOKEN_ID``, …) without changing the resolution machinery —
    each new kind is one enum member, one kwarg on
    :func:`nest_batch_by_mask`, and one entry in the dynamic-resolver
    dict inside :func:`make_mask_nesting_specs`.

    **Single-tokenizer assumption**: verl assumes all token-ID fields
    in a batch share the same tokenizer, so a single kwarg per
    enum member drives every sentinel-tagged entry. If you need
    per-field pad values (e.g. a distillation teacher with a different
    tokenizer), override that field explicitly via
    :func:`nest_batch_by_mask`'s ``field_to_mask_and_pad`` kwarg::

        nest_batch_by_mask(
            data,
            pad_token_id=student_tok.pad_token_id,
            field_to_mask_and_pad={
                "teacher_ids": ("attention_mask", teacher_tok.pad_token_id),
            },
        )
    """

    PAD_TOKEN_ID = enum.auto()

    def __repr__(self) -> str:
        return self.name


# Convenience alias for the common PAD_TOKEN_ID case — equivalent to
# ``DynamicPadValue.PAD_TOKEN_ID`` but less verbose at call sites.
PAD_TOKEN_ID = DynamicPadValue.PAD_TOKEN_ID


# Per-field nesting registry for verl's PPO pipeline.
#
# Each entry maps a data field name to ``(mask_field, pad_value)``:
# * ``mask_field`` — the name of the mask in the TensorDict that covers
#   this field's ``(*batch_dims, *sample_mask_dims)``.
# * ``pad_value`` — a literal value to fill at mask=False positions when
#   unnesting, or :data:`PAD_TOKEN_ID` to signal "pull this from
#   ``tokenizer.pad_token_id`` at spec-building time".
#
# Literal-pad fields have semantics that are stable across tokenizer /
# model: they are either (a) always consumed alongside a mask that zeroes
# out pad positions in loss/metric computations, or (b) semantically
# neutral at 0 / 0.0.
KNOWN_FIELD_TO_MASK_AND_PAD: dict[str, tuple[str, int | float | DynamicPadValue]] = {
    # paired with attention_mask (full prompt+response axis)
    "input_ids": ("attention_mask", PAD_TOKEN_ID),
    "teacher_ids": ("attention_mask", PAD_TOKEN_ID),
    "position_ids": ("attention_mask", 0),
    "old_log_probs": ("attention_mask", 0.0),
    "ref_log_prob": ("attention_mask", 0.0),
    "rollout_log_probs": ("attention_mask", 0.0),
    "entropys": ("attention_mask", 0.0),
    "teacher_logprobs": ("attention_mask", 0.0),
    "routed_experts": ("attention_mask", 0),
    # paired with response_mask (response-only axis)
    "advantages": ("response_mask", 0.0),
    "returns": ("response_mask", 0.0),
    "values": ("response_mask", 0.0),
    "token_level_scores": ("response_mask", 0.0),
    "token_level_rewards": ("response_mask", 0.0),
    "rm_scores": ("response_mask", 0.0),
    "rollout_is_weights": ("response_mask", 0.0),
    "sum_pi_squared": ("response_mask", 0.0),
}


# ---------------------------------------------------------------------------
# dtype compression
# ---------------------------------------------------------------------------

# Fields whose values fit a narrower dtype than the producer emitted
# (memory / RPC bandwidth saving). Each entry maps a field name to
# ``(target_dtype, valid_min, valid_max)``; :func:`compress_batch_dtypes`
# casts only when the live tensor's actual range fits in ``[valid_min,
# valid_max]`` — otherwise the cast would silently wrap around (most
# commonly when a producer uses a negative sentinel).
KNOWN_FIELD_DTYPE_COMPRESSIONS: dict[str, tuple[torch.dtype, int, int]] = {
    # MoE routing indices in verl fit comfortably in uint8 (≤ 256 experts).
    "routed_experts": (torch.uint8, 0, 255),
}


def compress_batch_dtypes(data: TensorDict) -> TensorDict:
    """Cast eligible fields down to narrower dtypes in place.

    Concerns orthogonal to mask nesting: this operation only rewrites
    field dtypes where the live values fit a smaller type, to save
    memory and RPC bandwidth. Call **before** :func:`nest_batch_by_mask`
    — once a field is nested, its values are hidden behind a
    ``NestedTensor`` and per-element range checks are more awkward.

    Uses :data:`KNOWN_FIELD_DTYPE_COMPRESSIONS` as the registry of
    eligible fields. Unlisted fields are left untouched. Already
    nested fields and tensors whose range doesn't fit the target
    dtype are silently skipped.

    Args:
        data: TensorDict to compress in-place.

    Returns:
        The mutated TensorDict.
    """
    for field, (target_dtype, valid_min, valid_max) in KNOWN_FIELD_DTYPE_COMPRESSIONS.items():
        t = data.get(field, None)
        if t is None or not isinstance(t, torch.Tensor) or t.is_nested:
            continue
        if t.dtype == target_dtype:
            continue
        if t.min() >= valid_min and t.max() <= valid_max:
            data[field] = t.to(target_dtype)
    return data


def make_mask_nesting_specs(
    data: TensorDict,
    pad_token_id: int | None = None,
    field_to_mask_and_pad: dict[str, tuple[str, int | float | bool | DynamicPadValue]] | None = None,
) -> dict[str, MaskNestingSpec]:
    """Build :class:`MaskNestingSpec` objects for a padded TensorDict.

    Classification is **purely by data-field name**, resolved against two
    sources merged in this order (later overrides earlier):

    1. :data:`KNOWN_FIELD_TO_MASK_AND_PAD` — the library-owned registry.
       Entries map ``field → (mask_field, pad_value)`` where ``pad_value``
       is either a literal (fully static) or a :class:`DynamicPadValue`
       member such as :data:`PAD_TOKEN_ID` (mask is known, pad value is
       dynamic and resolved from the matching kwarg).
    2. ``field_to_mask_and_pad`` — caller-supplied overrides, used for
       custom user fields or to override a dynamic entry individually
       (e.g. a distillation teacher with a different tokenizer:
       ``{"teacher_ids": ("attention_mask", teacher_pad_id)}``).

    Every :class:`DynamicPadValue` left unresolved after merging (i.e.
    the caller did not pass the matching kwarg) causes that field to be
    silently skipped — it is treated as "unknown" and left untouched.

    Unknown field names are silently ignored.

    The function groups data fields by their target mask and emits one
    :class:`MaskNestingSpec` per mask. Tensor shape is only used as a
    sanity check (``data.shape[: mask.ndim] == mask.shape``).

    Each data tensor must have shape
    ``(*batch_dims, *sample_mask_dims, *feature_dims)`` such that
    ``(*batch_dims, *sample_mask_dims) == mask.shape``. The leading dims
    must match the mask exactly; any trailing ``feature_dims`` (e.g.
    multi-head) pass through untouched.

    The returned specs are plain dataclasses — callers can freely mutate
    ``data_field_to_pad_value`` on them (add, remove, or override entries)
    before passing them to :func:`nest_batch`.

    Args:
        data: TensorDict containing the data and mask tensors.
        pad_token_id: Tokenizer pad token id. Resolves every
            :data:`PAD_TOKEN_ID` entry in the merged registry. A single
            global value is applied to all token-ID fields — see
            :class:`DynamicPadValue` for the rationale and how to opt
            out per-field.
        field_to_mask_and_pad: Caller-supplied per-field overrides,
            ``{field: (mask_field, pad_value)}``. Use this for custom
            user fields or to override a dynamic entry individually.

    Returns:
        ``{mask_field: MaskNestingSpec}`` — one entry per mask that has
        at least one data field bound to it.
    """
    merged: dict[str, tuple[str, int | float | bool | DynamicPadValue]] = dict(KNOWN_FIELD_TO_MASK_AND_PAD)
    if field_to_mask_and_pad:
        merged.update(field_to_mask_and_pad)

    # Dispatch table mapping each dynamic kind to its resolved value
    # (or None if the caller didn't supply it). Extend this dict when
    # adding new DynamicPadValue members.
    dynamic_resolved: dict[DynamicPadValue, int | None] = {
        DynamicPadValue.PAD_TOKEN_ID: pad_token_id,
    }

    # Group data fields by their target mask field.
    fields_by_mask: dict[str, dict[str, int | float | bool]] = {}
    for data_field in list(data.keys()):
        if data_field not in merged:
            continue  # unknown field, leave untouched
        t = data.get(data_field, None)
        if t is None or not isinstance(t, torch.Tensor) or t.is_nested:
            continue

        mask_field, pad_value = merged[data_field]

        # Resolve dynamic pad values from the dispatch table. Silently
        # skip the field if the caller did not supply the matching
        # kwarg (treat as "unknown field").
        if isinstance(pad_value, DynamicPadValue):
            resolved = dynamic_resolved.get(pad_value)
            if resolved is None:
                continue
            pad_value = resolved

        assert mask_field in data, (
            f"field {data_field!r} requires mask {mask_field!r} which is missing from the TensorDict"
        )
        mask_t = data[mask_field]
        # Validate that the field's leading dims (after any registered
        # permutation from KNOWN_FIELD_PERMUTATIONS) match the mask.
        # nest_batch_by_mask will apply the permutation for real.
        perm = _permutation_for_field(data_field, t.ndim)
        canonical_shape = tuple(t.shape[p] for p in perm) if perm is not None else tuple(t.shape)
        expected_prefix = tuple(mask_t.shape)
        if canonical_shape[: len(expected_prefix)] != expected_prefix:
            perm_hint = f" (after permutation {perm})" if perm is not None else ""
            raise AssertionError(
                f"field {data_field!r} has shape {tuple(t.shape)}{perm_hint}; "
                f"expected leading (*batch_dims, *sample_mask_dims) to match "
                f"mask {mask_field!r} shape {expected_prefix}. "
                f"Hint: if this is a new layout that needs a permutation, "
                f"add an entry to KNOWN_FIELD_PERMUTATIONS."
            )
        fields_by_mask.setdefault(mask_field, {})[data_field] = pad_value

    specs: dict[str, MaskNestingSpec] = {}
    for mask_field, fields in fields_by_mask.items():
        specs[mask_field] = MaskNestingSpec(
            mask_field=mask_field,
            mask_shape=tuple(data[mask_field].shape),
            data_field_to_pad_value=fields,
        )
    return specs


# Non-tensor keys where `nest_batch_by_mask` stashes reversal state so
# that `unnest_batch_by_mask` can fully reverse the operation without
# the caller having to thread anything through themselves.
_MASK_NESTING_SPECS_KEY = "_mask_nesting_specs"
_MASK_NESTING_PERMUTATIONS_KEY = "_mask_nesting_permutations"


# Per-field permutations that canonicalise a producer's layout into
# ``(*batch_dims, *sample_mask_dims, *feature_dims)`` before nesting.
#
# Classification is **explicit by field name** — no shape inference
# (same principle as :data:`KNOWN_FIELD_TO_MASK_AND_PAD`). Each entry
# maps a field name to a ``torch.permute`` tuple that the library
# applies iff the live tensor's rank matches the permutation's length.
# This lets a single entry handle the common "sometimes multimodal,
# sometimes not" case for ``position_ids``: 2-D ``(bs, seq_len)`` has
# ``ndim == 2 != 3`` so the registered 3-D permutation is skipped;
# 3-D ``(bs, heads, seq_len)`` matches and gets permuted to
# ``(bs, seq_len, heads)``.
#
# ``unnest_batch_by_mask`` reads the same permutations from stashed
# state and applies the inverse, so the round-trip is transparent to
# downstream code.
KNOWN_FIELD_PERMUTATIONS: dict[str, tuple[int, ...]] = {
    # Multimodal position_ids: (bs, heads, seq_len) → (bs, seq_len, heads).
    "position_ids": (0, 2, 1),
}


def _inverse_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    """Return the permutation that undoes *perm*."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def _permutation_for_field(field: str, ndim: int) -> tuple[int, ...] | None:
    """Return the registered permutation for *field* if it applies to a rank-*ndim* tensor.

    Returns ``None`` when the field is unlisted or the registered
    permutation's length does not match ``ndim`` (e.g. a 3-D
    ``position_ids`` permutation declared in the registry does not
    apply to a 2-D text-only ``position_ids`` tensor).
    """
    perm = KNOWN_FIELD_PERMUTATIONS.get(field)
    if perm is None or len(perm) != ndim:
        return None
    return perm


def nest_batch_by_mask(
    data: TensorDict,
    specs: dict[str, MaskNestingSpec] | None = None,
    *,
    pad_token_id: int | None = None,
    field_to_mask_and_pad: dict[str, tuple[str, int | float | bool | DynamicPadValue]] | None = None,
) -> TensorDict:
    """Nest every eligible tensor field in *data* in place using mask-driven RLE.

    There are three ways to drive the nesting, listed from simplest to
    most flexible:

    1. **Simple** — let the library resolve specs from
       :data:`KNOWN_FIELD_TO_MASK_AND_PAD` and the tokenizer's pad id::

           nest_batch_by_mask(data, pad_token_id=tokenizer.pad_token_id)

    2. **Declarative custom fields** — add or override entries via the
       ``field_to_mask_and_pad`` kwarg::

           nest_batch_by_mask(
               data,
               pad_token_id=tokenizer.pad_token_id,
               field_to_mask_and_pad={"my_field": ("attention_mask", 0.0)},
           )

    3. **Dynamic / programmatic** — build specs with
       :func:`make_mask_nesting_specs`, mutate them, and pass them back::

           specs = make_mask_nesting_specs(data, pad_token_id=tok.pad_token_id)
           specs["attention_mask"].data_field_to_pad_value.pop("routed_experts", None)
           nest_batch_by_mask(data, specs=specs)

    Before invoking ``spec.nest_in_td``, each tracked field whose
    layout is registered in :data:`KNOWN_FIELD_PERMUTATIONS` is
    permuted to the canonical
    ``(*batch_dims, *sample_mask_dims, *feature_dims)`` form. This
    transparently handles producers that emit a non-canonical layout
    — most notably multimodal 3-D ``position_ids``
    ``(bs, heads, seq_len)`` which is auto-permuted to
    ``(bs, seq_len, heads)`` before nesting and permuted back by
    :func:`unnest_batch_by_mask`.

    The resolved specs and any permutations applied are stashed as
    non-tensor data so :func:`unnest_batch_by_mask` can fully reverse
    the operation without the caller re-threading anything.

    Args:
        data: TensorDict to nest in-place.
        specs: Pre-built ``{mask_field: MaskNestingSpec}`` for path 3.
            Mutually exclusive with ``pad_token_id`` /
            ``field_to_mask_and_pad``.
        pad_token_id: Tokenizer pad token id (paths 1 & 2).
        field_to_mask_and_pad: Per-field overrides (path 2).

    Returns:
        The mutated TensorDict.
    """
    if specs is None:
        specs = make_mask_nesting_specs(data, pad_token_id=pad_token_id, field_to_mask_and_pad=field_to_mask_and_pad)
    elif pad_token_id is not None or field_to_mask_and_pad is not None:
        raise TypeError(
            "nest_batch_by_mask: pass either `specs` (pre-built) or the "
            "`pad_token_id` / `field_to_mask_and_pad` kwargs, not both."
        )

    permutations: dict[str, tuple[int, ...]] = {}
    for spec in specs.values():
        for field in spec.data_field_to_pad_value:
            t = data[field]
            perm = _permutation_for_field(field, t.ndim)
            if perm is not None:
                data[field] = t.permute(perm).contiguous()
                permutations[field] = perm

    # Apply specs in a deterministic order (insertion order of the dict).
    for spec in specs.values():
        spec.nest_in_td(data)

    # Stash reversal state.
    tu.assign_non_tensor_data(data, _MASK_NESTING_SPECS_KEY, specs)
    if permutations:
        tu.assign_non_tensor_data(data, _MASK_NESTING_PERMUTATIONS_KEY, permutations)

    return data


def response_from_nested(tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Extract response from nested model output.

    Args:
        tensor: a nested tensor with shape (bsz, prompt_len + response_len)
        response_mask: a nested tensor with shape (bsz, response_len)

    Returns:
        tensor: a nested tensor with shape (bsz, response_len)
    """
    values, offsets = tensor.values(), tensor.offsets()
    response_lens = response_mask.offsets().diff()
    response_list = []
    for resp_len, seq_offset in zip(response_lens, offsets[1:], strict=True):
        # left-shift model output by one token for log_probs/values
        response_list.append(values[seq_offset - resp_len - 1 : seq_offset - 1])
    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)


def response_to_nested(tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Convert padded response tensor to nested tensor.

    Args:
        tensor: a tensor with shape (bsz, response_len)
        response_mask: a nested tensor with shape (bsz, response_len)

    Returns:
        tensor: a nested tensor with shape (bsz, response_len)
    """
    assert response_mask.is_nested
    response_lens = response_mask.offsets().diff()
    response_list = []
    for i in range(tensor.shape[0]):
        response_list.append(tensor[i, : response_lens[i]])

    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)


def unnest_batch_by_mask(
    data: TensorDict,
    specs: dict[str, MaskNestingSpec] | None = None,
) -> TensorDict:
    """Inverse of :func:`nest_batch_by_mask`. Reconstruct padded fields in place.

    When ``specs`` is not supplied, retrieves the ones that
    :func:`nest_batch_by_mask` stashed under ``_mask_nesting_specs``.
    If the TensorDict was never nested (key absent) this is a no-op.

    Converts every nested tensor tracked by *specs* back to dense
    padded form, restores the corresponding mask fields, and reverses
    any shape-normalising permutations applied on the way in.
    Trailing ``feature_dims`` (e.g. multi-head) are preserved
    automatically by ``MaskNestingSpec.unnest_in_td``.

    Args:
        data: TensorDict previously transformed by :func:`nest_batch_by_mask`.
        specs: Optional ``{mask_field: MaskNestingSpec}``. When ``None``,
            retrieved from ``data``'s stashed state.

    Returns:
        The same TensorDict with dense padded tensors restored.
    """
    if specs is None:
        specs = tu.get_non_tensor_data(data, _MASK_NESTING_SPECS_KEY, default=None)
        if specs is None:
            # Never nested — nothing to undo.
            return data

    for spec in specs.values():
        spec.unnest_in_td(data)

    # Reverse any shape-normalising permutations applied on the way in.
    permutations = tu.get_non_tensor_data(data, _MASK_NESTING_PERMUTATIONS_KEY, default=None)
    if permutations:
        for field, perm in permutations.items():
            inv = _inverse_permutation(perm)
            data[field] = data[field].permute(inv).contiguous()

    # Clean up stashed state.
    data.pop(_MASK_NESTING_SPECS_KEY, None)
    data.pop(_MASK_NESTING_PERMUTATIONS_KEY, None)

    return data


# ---------------------------------------------------------------------------
# Layer 1: library-level nested → dense unnest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnnestContext:
    """Precomputed scatter indices for :func:`unnest`."""

    row_idx: torch.Tensor
    col_idx: torch.Tensor
    batch_size: int
    seq_len: int


def prepare_unnest(data: TensorDict) -> UnnestContext:
    """Precompute scatter indices from a mask-nested TensorDict's RLE."""
    offsets = data["attention_mask_offsets"]
    lengths = data["attention_mask_lengths"]
    row_idx, col_idx, _, seq_len = _rle_scatter_indices(offsets, lengths)
    return UnnestContext(
        row_idx=row_idx,
        col_idx=col_idx,
        batch_size=offsets.shape[0],
        seq_len=int(seq_len),
    )


def unnest(ctx: UnnestContext, tensor: torch.Tensor) -> torch.Tensor:
    """Scatter a nested tensor into a zero-filled dense ``(batch_size, seq_len, *trailing)``."""
    data_flat = tensor.values() if tensor.is_nested else tensor
    trailing_shape = data_flat.shape[1:]
    dense = torch.full(
        (ctx.batch_size, ctx.seq_len, *trailing_shape),
        0.0,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if ctx.row_idx.numel() > 0:
        dense[ctx.row_idx, ctx.col_idx] = data_flat
    return dense


# ---------------------------------------------------------------------------
# Layer 2: PPO-level response slicing from dense (bs, seq_len, *) tensors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseSliceContext:
    """Precomputed per-sample slice bounds for :func:`slice_response`.

    Bounds bake in the PPO one-token left-shift.
    """

    slice_bounds: list[tuple[int, int, int]]  # one (resp_start, resp_end, pad_size) per sample
    max_response_len: int


def prepare_response_slice(data: TensorDict) -> ResponseSliceContext:
    """Derive per-sample ``(resp_start, resp_end, pad_size)`` bounds from a mask-nested batch.

    Uses ``prompts`` / ``responses`` for lengths and the RLE's first
    offset per row for absolute positions. Original padded dimensions
    are recovered from stashed :class:`MaskNestingSpec` when available,
    falling back to live masks otherwise.
    """
    offsets = data["attention_mask_offsets"]
    lengths = data["attention_mask_lengths"]
    batch_size = offsets.shape[0]

    # Prefer stashed spec for original dims; fall back to live mask.
    specs = tu.get_non_tensor_data(data, _MASK_NESTING_SPECS_KEY, default={}) or {}

    def _orig_mask_len(mask_field: str) -> int | None:
        if mask_field in specs:
            return specs[mask_field].mask_shape[-1]
        if mask_field in data:
            return data[mask_field].shape[-1]
        return None

    orig_seq_len = _orig_mask_len("attention_mask")
    orig_response_len = _orig_mask_len("response_mask")

    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        max_response_len = orig_response_len if orig_response_len is not None else response_lens.max().item()
    else:
        mask_shape = (batch_size, orig_seq_len) if orig_seq_len is not None else None
        attn_mask = rle_to_mask(offsets, lengths, shape=mask_shape)
        prompt_lens = attn_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attn_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = orig_response_len if orig_response_len is not None else response_ids.shape[1]

    rle_first_offsets = offsets.values()[offsets._offsets[:-1]]
    slice_bounds: list[tuple[int, int, int]] = []
    for i in range(batch_size):
        abs_start = rle_first_offsets[i].item()
        p_len = prompt_lens[i].item()
        r_len = response_lens[i].item()
        # left-shift by 1 for log_prob/value alignment
        resp_start = abs_start + p_len - 1
        resp_end = abs_start + p_len + r_len - 1
        pad_size = max_response_len - r_len
        slice_bounds.append((resp_start, resp_end, pad_size))

    return ResponseSliceContext(slice_bounds=slice_bounds, max_response_len=int(max_response_len))


def slice_response(ctx: ResponseSliceContext, dense: torch.Tensor) -> torch.Tensor:
    """Slice the response region from a dense ``(bs, seq_len, *trailing)`` tensor."""
    trailing_shape = dense.shape[2:]
    skip_padding = (0, 0) * len(trailing_shape)
    response_list = [
        F.pad(dense[i, start:end], (*skip_padding, 0, pad)) for i, (start, end, pad) in enumerate(ctx.slice_bounds)
    ]
    return torch.stack(response_list, dim=0)


# ---------------------------------------------------------------------------
# Dispatching layer: unified response extraction over both batch nesting styles
# ---------------------------------------------------------------------------


def extract_response(data: TensorDict, tensor: torch.Tensor) -> torch.Tensor:
    """Extract the padded response region from a model output tensor.

    Dispatches on batch nesting style: new path (``attention_mask_offsets``
    present) runs :func:`unnest` → :func:`slice_response`; legacy path
    slices the flat ``(total_nnz, *)`` values directly. Both apply the
    PPO one-token left-shift and right-pad to ``max_response_len``.
    """
    if "attention_mask_offsets" in data:
        return slice_response(prepare_response_slice(data), unnest(prepare_unnest(data), tensor))

    # Legacy path — mirrors no_padding_2_padding inline.
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_response_len < 0:
            max_response_len = response_lens.max().item()
    else:
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]
    assert not prompt_lens.eq(0).any(), (
        f"flat_start = seq_offset - resp_len - 1 assumes prompt_len > 0. Got {prompt_lens}"
    )

    skip_padding = (0, 0) * (values.ndim - 1)
    response_list = []
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        # left-shift by 1 for log_prob/value alignment
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (*skip_padding, 0, pad_size)))
    return torch.stack(response_list, dim=0)
