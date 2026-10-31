# Copyright 2025 Individual Contributor: Deepro Choudhury
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
"""Shared-prefix packing for the no-padding micro-batch layout.

Adjacent samples with an identical prompt are packed as one trunk plus one branch per sample::

    [prompt[:-1]] [prompt[-1] + response_1] ... [prompt[-1] + response_n]

Each branch repeats the last prompt token because ``no_padding_2_padding`` reads the first
response log-prob from that row. A branch attends to ``[trunk; branch]`` through one varlen
flash-attention call, so the result equals the unpacked forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from tensordict import TensorDict
from transformers import PretrainedConfig
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask

from verl.utils.device import is_cuda_available
from verl.utils.model import extract_multi_modal_inputs

__all__ = [
    "PrefixPackingAttnMeta",
    "PrefixPackingPlan",
    "find_prefix_groups",
    "build_prefix_packing_plan",
    "packed_group_lens",
    "split_groups_to_budget",
    "check_prefix_packing_supported",
    "check_prefix_packing_micro_batch",
    "check_layers_seen",
]


class _LayerCounter:
    # A plain object, not a dataclass field value: FSDP copies dataclass kwargs with
    # ``dataclasses.replace`` when moving them to the device, and the copy must share the count.
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0


@dataclass
class PrefixPackingAttnMeta:
    """Attention metadata of one packed micro-batch, passed to the attention wrapper as a kwarg.

    Args:
        kv_index: ``(total_k,)`` packed-row ids; segment ``s`` attends to
            ``kv_index[cu_seqlens_k[s]:cu_seqlens_k[s + 1]]``.
        cu_seqlens_q: ``(num_segments + 1,)`` int32 cumulative query lengths.
        cu_seqlens_k: ``(num_segments + 1,)`` int32 cumulative key lengths.
        max_seqlen_q: Longest query segment.
        max_seqlen_k: Longest key segment.
        num_rows: Packed rows, including any tail pad.
    """

    kv_index: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    num_rows: int
    counter: _LayerCounter = field(default_factory=_LayerCounter, repr=False, compare=False)

    @property
    def layers_seen(self) -> int:
        """Number of attention layers that took the packed path in the last forward."""
        return self.counter.value

    def record_layer(self) -> None:
        """Count one attention layer; called by the attention wrapper."""
        self.counter.value += 1

    def with_tail_pad(self, pad_size: int) -> PrefixPackingAttnMeta:
        """Return a copy with ``pad_size`` trailing rows that attend only to themselves."""
        if pad_size <= 0:
            return self
        pad_rows = torch.arange(self.num_rows, self.num_rows + pad_size, device=self.kv_index.device)
        return replace(
            self,
            kv_index=torch.cat([self.kv_index, pad_rows]),
            cu_seqlens_q=torch.cat([self.cu_seqlens_q, self.cu_seqlens_q[-1:] + pad_size]),
            cu_seqlens_k=torch.cat([self.cu_seqlens_k, self.cu_seqlens_k[-1:] + pad_size]),
            max_seqlen_q=max(self.max_seqlen_q, pad_size),
            max_seqlen_k=max(self.max_seqlen_k, pad_size),
            num_rows=self.num_rows + pad_size,
            counter=_LayerCounter(),
        )


@dataclass
class PrefixPackingPlan:
    """Pack and unpack indices of one micro-batch.

    Args:
        attn: Attention metadata of the packed rows.
        pack_index: ``packed = flat[pack_index]`` for ids, positions and temperature.
        unpack_index: ``unpacked = packed_output[unpack_index]`` for per-token outputs.
        num_unpacked: Tokens before packing.
        num_packed: Tokens after packing.
    """

    attn: PrefixPackingAttnMeta
    pack_index: torch.Tensor
    unpack_index: torch.Tensor
    num_unpacked: int
    num_packed: int


def find_prefix_groups(input_ids: torch.Tensor, prompt_lens: torch.Tensor) -> tuple[list[int], list[int], list[int]]:
    """Group adjacent samples whose prompts are identical.

    Samples are never reordered. A prompt shorter than two tokens is never shared.

    Args:
        input_ids: Jagged nested tensor of token ids.
        prompt_lens: ``(batch,)`` prompt lengths.

    Returns:
        ``(group_sizes, seq_lens, prompt_lens)`` as Python lists.
    """
    flat = input_ids.values()
    offsets = input_ids.offsets().to(torch.int64)
    batch_size = prompt_lens.numel()
    prompt_lens = prompt_lens.to(device=flat.device, dtype=torch.int64)
    host = torch.cat([offsets, prompt_lens]).tolist()
    offsets_l, prompt_lens_l = host[: batch_size + 1], host[batch_size + 1 :]
    seq_lens_l = [end - start for start, end in zip(offsets_l[:-1], offsets_l[1:], strict=True)]
    assert all(p <= s for p, s in zip(prompt_lens_l, seq_lens_l, strict=True)), "prompt longer than sample"
    if batch_size == 0:
        return [], seq_lens_l, prompt_lens_l

    max_prompt = max(prompt_lens_l)
    if batch_size > 1 and max_prompt >= 2:
        pos = torch.arange(max_prompt, device=flat.device)
        toks = flat[(offsets[:-1, None] + pos).clamp_(max=flat.numel() - 1)]
        in_prompt = pos < prompt_lens[:, None]
        same_toks = ((toks[1:] == toks[:-1]) | ~in_prompt[1:]).all(dim=-1)
        joinable = ((prompt_lens[1:] == prompt_lens[:-1]) & same_toks & (prompt_lens[1:] >= 2)).tolist()
    else:
        joinable = [False] * (batch_size - 1)

    group_sizes = [1]
    for join in joinable:
        if join:
            group_sizes[-1] += 1
        else:
            group_sizes.append(1)
    return group_sizes, seq_lens_l, prompt_lens_l


def build_prefix_packing_plan(input_ids: torch.Tensor, prompt_lens: torch.Tensor) -> PrefixPackingPlan:
    """Build the packing plan of one micro-batch.

    Args:
        input_ids: Jagged nested tensor of token ids.
        prompt_lens: ``(batch,)`` prompt lengths.

    Returns:
        The plan. Without any shared prompt it reproduces the unpacked layout.
    """
    device = input_ids.offsets().device
    group_sizes, seq_lens, prompt_lens_l = find_prefix_groups(input_ids, prompt_lens)
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)

    pack, unpack, kv, q_lens, k_lens = [], [], [], [], []
    cursor = 0

    def add_segment(src_start: int, length: int, keys_before: list[torch.Tensor]) -> torch.Tensor:
        nonlocal cursor
        rows = torch.arange(cursor, cursor + length)
        pack.append(torch.arange(src_start, src_start + length))
        kv.extend([*keys_before, rows])
        q_lens.append(length)
        k_lens.append(length + sum(k.numel() for k in keys_before))
        cursor += length
        return rows

    row = 0
    for size in group_sizes:
        trunk_len = prompt_lens_l[row] - 1 if size > 1 else 0
        trunk = add_segment(offsets[row], trunk_len, []) if trunk_len else None
        for sample in range(row, row + size):
            keys_before = [trunk] if trunk is not None else []
            branch = add_segment(offsets[sample] + trunk_len, seq_lens[sample] - trunk_len, keys_before)
            unpack.extend([*keys_before, branch])
        row += size

    def cat(chunks: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(chunks).to(device) if chunks else torch.empty(0, dtype=torch.int64, device=device)

    attn = PrefixPackingAttnMeta(
        kv_index=cat(kv),
        cu_seqlens_q=torch.tensor([0] + q_lens).cumsum(0).to(device=device, dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0] + k_lens).cumsum(0).to(device=device, dtype=torch.int32),
        max_seqlen_q=max(q_lens, default=0),
        max_seqlen_k=max(k_lens, default=0),
        num_rows=cursor,
    )
    return PrefixPackingPlan(
        attn=attn, pack_index=cat(pack), unpack_index=cat(unpack), num_unpacked=offsets[-1], num_packed=cursor
    )


def packed_group_lens(group_sizes: list[int], seq_lens: list[int], prompt_lens: list[int]) -> list[int]:
    """Token count of each group after packing.

    Args:
        group_sizes: Group sizes in batch order.
        seq_lens: Per-sample sequence lengths.
        prompt_lens: Per-sample prompt lengths.

    Returns:
        One packed length per group.
    """
    lens, row = [], 0
    for size in group_sizes:
        total = sum(seq_lens[row : row + size])
        lens.append(total - (size - 1) * (prompt_lens[row] - 1) if size > 1 else total)
        row += size
    return lens


def split_groups_to_budget(
    group_sizes: list[int], seq_lens: list[int], prompt_lens: list[int], max_token_len: int
) -> list[int]:
    """Split groups whose packed length exceeds ``max_token_len``; each part keeps its own trunk.

    Args:
        group_sizes: Group sizes in batch order.
        seq_lens: Per-sample sequence lengths.
        prompt_lens: Per-sample prompt lengths.
        max_token_len: Token budget of one micro-batch.

    Returns:
        Group sizes in batch order, with the same total.
    """
    out, row = [], 0
    for size in group_sizes:
        trunk_len = prompt_lens[row] - 1
        count, length = 0, trunk_len
        for sample in range(row, row + size):
            branch_len = seq_lens[sample] - trunk_len
            if count and length + branch_len > max_token_len:
                out.append(count)
                count, length = 0, trunk_len
            count += 1
            length += branch_len
        out.append(count)
        row += size
    return out


def check_prefix_packing_supported(
    config: PretrainedConfig, *, use_remove_padding: bool, ulysses_sp_size: int, pass_packed_cu_seqlens: bool
) -> None:
    """Raise if ``use_prefix_packing=True`` cannot work with this model and engine setup.

    Args:
        config: Hugging Face config of the model.
        use_remove_padding: Whether the engine packs sequences.
        ulysses_sp_size: Ulysses sequence-parallel size.
        pass_packed_cu_seqlens: Whether the model takes ``cu_seqlens`` directly.

    Raises:
        ValueError: If any requirement is not met.
    """
    problems = []
    if not use_remove_padding:
        problems.append("use_remove_padding=False (packing works on the no-padding path)")
    if ulysses_sp_size > 1:
        problems.append("Ulysses sequence parallelism")
    attn_impl = getattr(config, "_attn_implementation", None)
    if attn_impl not in ("flash_attention_2", "flash_attention_3"):
        problems.append(f"attn_implementation={attn_impl!r} (needs flash_attention_2 or flash_attention_3)")
    if hasattr(config, "vision_config"):
        problems.append("vision-language models")
    if pass_packed_cu_seqlens:
        problems.append("models that take cu_seqlens directly, such as linear-attention hybrids")
    if not is_cuda_available:
        problems.append("non-CUDA devices")
    elif flash_attn_supports_top_left_mask():
        problems.append("flash-attn < 2.1 (needs bottom-right causal alignment)")
    if problems:
        raise ValueError(
            f"use_prefix_packing=True is not supported with {'; '.join(problems)}. Disable use_prefix_packing."
        )


def check_prefix_packing_micro_batch(
    micro_batch: TensorDict, *, distillation_use_topk: bool, score_centering: bool
) -> None:
    """Raise if a micro-batch carries per-token inputs that cannot follow the packed layout.

    Args:
        micro_batch: The micro-batch about to be packed.
        distillation_use_topk: Whether top-k distillation is on.
        score_centering: Whether score centering is on.

    Raises:
        RuntimeError: If packing is not supported for this batch.
    """
    problems = []
    if distillation_use_topk or score_centering:
        problems.append("top-k distillation or score centering")
    if micro_batch.get("routed_experts", None) is not None:
        problems.append("router replay")
    if micro_batch["position_ids"].dim() == 3 or extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", [])):
        problems.append("multi-modal inputs")
    if micro_batch.get("prompts", None) is None:
        problems.append("batches without 'prompts'")
    if problems:
        raise RuntimeError(
            f"use_prefix_packing=True is not supported with {'; '.join(problems)}: "
            "their per-token layout cannot follow the packed row. Disable use_prefix_packing."
        )


def check_layers_seen(meta: PrefixPackingAttnMeta, num_layers: int) -> None:
    """Raise if some attention layer bypassed the packed attention and so mixed the branches.

    Args:
        meta: Metadata used in the forward.
        num_layers: Number of attention layers of the model.

    Raises:
        RuntimeError: If fewer than ``num_layers`` layers took the packed path.
    """
    if meta.layers_seen != num_layers:
        raise RuntimeError(
            f"use_prefix_packing=True needs every attention layer to use verl's flash-attention wrapper, but "
            f"{meta.layers_seen}/{num_layers} did, so the branches would attend to each other. "
            "Disable use_prefix_packing for this model."
        )
