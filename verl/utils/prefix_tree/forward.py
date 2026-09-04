# Copyright 2025 Meituan Ltd. and/or its affiliates
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
"""Prefix-tree forward-path: tree build, MAGI dispatch, rope override,
    fused/unfused forward drivers, and LCE post-processing.

Public: prepare_prefix_tree, tree_post_processing, prefix_tree_output_processor, dispatch_magi."""

from __future__ import annotations

import logging as _log
from collections import Counter, OrderedDict, namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as _dist
from magi_attention.api import (
    DistAttnConfig,
    get_position_ids,
    magi_attn_flex_key,
    undispatch,
)
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta.solver.dispatch_solver import DispatchConfig
from megatron.core import parallel_state as mpu
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask

from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO
from verl.utils.prefix_tree.magi import (
    PackRestorationParam,
    PrefixTreeMagiBatch,
    build_prefix_tree_micro_batch,
    clear_rope_pids,
    prefix_tree_decoder_key_context,
    restore_flat_to_nested,
    set_rope_pids,
    strip_prefix_tree_args,
)

_logger = _log.getLogger(__name__)

TreeForwardCtx = namedtuple("TreeForwardCtx", ["pb", "input_ids", "position_ids", "attention", "model"])
"""Returned by :func:`prepare_prefix_tree`.  ``rope_exit`` is a
callable to deactivate the MAGI rope context (``None`` for flex / non-tree)."""

# Shared helpers


def _prepare_attn_inputs(
    pb: PrefixTreeMagiBatch,
    prefix_tree_attention: str,
) -> tuple[Tensor, Tensor, dict]:
    """Build local_input_ids, local_position_ids, and attention kwargs.

    MAGI path returns CP-local slices; flex returns full tree-packed."""
    if prefix_tree_attention == "magi":
        local_input_ids, local_position_ids = dispatch_magi(pb)
        attn_kwargs = {"magi_attention_key": pb.magi_key}
    else:
        if pb.flex_key is None:
            raise RuntimeError("flex attention requires pb.flex_key")
        local_input_ids = pb.tree_packed_input_ids.unsqueeze(0)
        local_position_ids = pb.tree_packed_position_ids.unsqueeze(0)
        attn_kwargs = {"flex_attention_key": pb.flex_key}
    return local_input_ids, local_position_ids, attn_kwargs


def _unpack_nested_to_list(x, mask: Optional[Tensor] = None) -> Optional[list[Tensor]]:
    """Unpack NestedTensor or padded 2-D Tensor into list of 1-D tensors. Returns None if cannot safely unpack."""
    if x is None:
        return None
    if hasattr(x, "is_nested") and x.is_nested:
        offsets = x.offsets()
        lengths = offsets.diff().tolist()
        vals = x.values()
        out: list[Tensor] = []
        pos = 0
        for length in lengths:
            out.append(vals[pos : pos + int(length)])
            pos += int(length)
        return out
    if x.dim() == 2 and mask is not None:
        seqlens = mask.sum(dim=-1).tolist()
        return [x[i, : int(seqlens[i])] for i in range(x.shape[0])]
    return None


def _build_flex_key(params, device, subtrie=None):
    """Build flex_attention block_mask from the trie topology.

    A query at flat position q sees a kv position iff kv's node is an ancestor-or-self
    of q's node (full), with same-node pairs causal. Interval containment via DFS
    enter/exit times; BFS flat layout puts ancestors strictly before descendants, so
    ``kv_idx <= q_idx`` binds same-node causality and is auto-true for ancestors.
    """
    if subtrie is None:
        raise RuntimeError("_build_flex_key: subtrie required for flex (trie topology)")

    # DFS enter/exit times: u is ancestor-or-self of v iff tin[u] <= tin[v] <= tout[u].
    tin: dict[int, int] = {}
    tout: dict[int, int] = {}
    counter = 0

    def _visit(node) -> None:
        nonlocal counter
        tin[node.node_idx] = counter
        counter += 1
        for child in subtrie.children_of(node):
            _visit(child)
        tout[node.node_idx] = counter
        counter += 1

    for root in subtrie.roots:
        _visit(root)

    # Map each flat position (BFS layout, same as utils.py Pass 1) to its node's interval.
    total = params.total_seqlen_q
    tin_arr = torch.zeros(total, dtype=torch.int64, device=device)
    tout_arr = torch.zeros(total, dtype=torch.int64, device=device)
    pos = 0
    for node in subtrie.bfs():
        length = len(node.input_ids)
        tin_arr[pos : pos + length] = tin[node.node_idx]
        tout_arr[pos : pos + length] = tout[node.node_idx]
        pos += length

    def prefix_tree_mask(b, h, q_idx, kv_idx):
        ancestor_or_self = (tin_arr[kv_idx] <= tin_arr[q_idx]) & (tout_arr[q_idx] <= tout_arr[kv_idx])
        return ancestor_or_self & (kv_idx <= q_idx)

    # _compile=False: avoid Triton JIT which takes minutes for new shapes.
    # Memory is handled at the call site via torch.utils.checkpoint.
    block_mask = create_block_mask(
        prefix_tree_mask, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device, _compile=False
    )
    block_mask._flex_aux = (tin_arr, tout_arr)  # pin closure tensors against GC
    return block_mask


def _build_magi_key(model, params):
    """Build magi_attn_flex_key from PrefixTreeParams and model config."""
    # TP shards heads: each rank holds heads/tp_size. GQA falls back to num_attention_heads.
    cfg = unwrap_model(model).config
    tp_size = mpu.get_tensor_model_parallel_world_size()
    num_heads_q = cfg.num_attention_heads // tp_size
    num_query_groups = getattr(cfg, "num_query_groups", cfg.num_attention_heads) or cfg.num_attention_heads
    num_heads_kv = max(1, num_query_groups // tp_size)
    head_dim = cfg.kv_channels  # hidden_size // num_attention_heads

    try:
        cp_group = mpu.get_context_parallel_group()
    except Exception:
        cp_group = _dist.group.WORLD

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges(params.q_ranges),
        k_ranges=AttnRanges.from_ranges(params.k_ranges),
        attn_mask_type=[AttnMaskType(m) for m in params.mask_types],
        total_seqlen_q=params.total_seqlen_q,
        total_seqlen_k=params.total_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(
            dispatch_config=DispatchConfig(uneven_shard=True),
        ),
    )


def _finalize_prefix_tree_batch(
    params,
    model,
    num_samples: int,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
    subtrie=None,
) -> PrefixTreeMagiBatch:
    """Pad to TP/CP divisibility, build attention key, and wrap into PrefixTreeMagiBatch."""
    real_tokens = params.tree_packed_tokens.shape[0]
    align_size = (tp_size * cp_size * 2) if cp_size > 1 else tp_size
    if align_size > 1:
        pad_len = (align_size - real_tokens % align_size) % align_size
        if pad_len > 0:
            params.tree_packed_tokens = torch.cat(
                [params.tree_packed_tokens, params.tree_packed_tokens.new_zeros(pad_len)]
            )
            params.tree_packed_position_ids = torch.cat(
                [params.tree_packed_position_ids, params.tree_packed_position_ids.new_zeros(pad_len)]
            )
            params.total_seqlen_q += pad_len
            params.total_seqlen_k += pad_len

    if attention_type == "magi":
        # Cache the MAGI key on the subtrie: OLP and actor_update process the same
        # micro-batch (same sequences, same seqlen) so the key is valid for both passes.
        # TODO(dynamic-cp): if dynamic_context_parallel is enabled, dump this cache.
        if subtrie is not None and getattr(subtrie, "_cached_magi_key", None) is not None:
            magi_key = subtrie._cached_magi_key
        else:
            magi_key = _build_magi_key(model, params)
            if subtrie is not None:
                subtrie._cached_magi_key = magi_key
        flex_key = None
    elif attention_type == "flex":
        flex_key = _build_flex_key(params, params.tree_packed_tokens.device, subtrie=subtrie)
        magi_key = None
    else:
        raise ValueError(f"Unknown attention_type: {attention_type!r} (expected 'magi' or 'flex')")

    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=magi_key,
        flex_key=flex_key,
        restoration=PackRestorationParam(
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            ancestor_segment_ranges=getattr(params, "_leaf_ancestor_ranges", None),
            boundary_registry=getattr(params, "boundary_registry", None),
        ),
        subtrie=subtrie,
        real_tokens=real_tokens,
    )


def dispatch_magi(pt_batch: PrefixTreeMagiBatch) -> tuple[Tensor, Tensor]:
    """Slice local_input_ids / local_position_ids via magi dispatch
    (get_position_ids). Each CP rank processes its assigned slice."""
    local_indices = get_position_ids(pt_batch.magi_key)
    local_input_ids = pt_batch.tree_packed_input_ids[local_indices].unsqueeze(0)
    local_position_ids = pt_batch.tree_packed_position_ids[local_indices].unsqueeze(0)
    return local_input_ids, local_position_ids


def build_prefix_tree_batch(model, input_ids, logits_processor_args):
    """Build prefix-tree micro-batch from logits_processor_args. Returns PrefixTreeMagiBatch or None."""
    args = logits_processor_args or {}
    prefix_tree_attention = args.get("prefix_tree_attention", "flex")
    loss_mask_nested = args.get("loss_mask")
    position_ids_nested = args.get("position_ids")
    # Per-mb subtrie from prepare_prefix_tree_micro_batches (global trie pruned to mb).
    subtrie = args.get("prefix_tree_subtree")

    return build_prefix_tree_micro_batch(
        model,
        input_ids,
        loss_mask_nested,
        position_ids=position_ids_nested,
        attention_type=prefix_tree_attention,
        tp_size=mpu.get_tensor_model_parallel_world_size(),
        cp_size=mpu.get_context_parallel_world_size(),
        subtrie=subtrie,
    )


def prepare_prefix_tree(
    model,
    input_ids,
    logits_processor_args,
    model_kwargs,
    *,
    vision_model=False,
    mtp_enable_train=False,
):
    """Prepare prefix-tree forward context.

    Returns a :class:`TreeForwardCtx` or ``None`` when tree is not applicable.
    On success, merges attention kwargs into *model_kwargs* in-place and (for
    MAGI) activates the rope context via ``ctx.rope_exit``.  The caller must
    deactivate rope via ``ctx.rope_exit(None, None, None)`` after
    post-processing (or on the intermediate-PP path).
    """
    if vision_model or mtp_enable_train:
        _logger.warning(
            "prefix_tree: skipping prefix-tree path (vision_model=%s, mtp_enable_train=%s), not fully supported yet",
            vision_model,
            mtp_enable_train,
        )
        strip_prefix_tree_args(logits_processor_args)
        return None

    prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")

    pb = build_prefix_tree_batch(model, input_ids, logits_processor_args)
    if pb is None:
        _logger.warning("prefix_tree: build_prefix_tree_batch returned None; falling back to standard THD path")
        strip_prefix_tree_args(logits_processor_args)
        return None

    local_input_ids, local_position_ids, attn_kwargs = _prepare_attn_inputs(pb, prefix_tree_attention)
    strip_prefix_tree_args(logits_processor_args)
    model_kwargs.update(attn_kwargs)

    set_rope_pids(model, local_position_ids)

    return TreeForwardCtx(pb, local_input_ids, local_position_ids, prefix_tree_attention, model)


def tree_post_processing(ctx, output_orig, logits_processor, logits_processor_args, post_process):
    if ctx is None:
        return output_orig
    pt_batch = ctx.pb
    real_tokens = pt_batch.real_tokens
    prefix_tree_attention = ctx.attention

    # Normalize to batch-first [1, seq, ...]. flex strips padding to real_tokens
    # here; magi keeps full local tokens and strips padding after undispatch.
    if output_orig.shape[0] == 1:
        if prefix_tree_attention != "magi":
            output_orig = output_orig[:, :real_tokens]
    else:
        if prefix_tree_attention != "magi":
            output_orig = output_orig[:real_tokens]
        output_orig = output_orig.permute(1, 0, 2)

    if not post_process or logits_processor is None:
        return output_orig.permute(1, 0, 2)

    try:
        logits_flat = output_orig.squeeze(0)

        orig_args = logits_processor_args or {}
        flat_args = {
            k: v for k, v in orig_args.items() if k not in ("label", "temperature", "loss_mask", "use_prefix_tree")
        }

        # undispatch from magi's cp dispatching
        # flex should not have cp anyway
        if prefix_tree_attention == "magi":
            logits_flat = undispatch(logits_flat, pt_batch.magi_key)[:real_tokens]

        n = len(pt_batch.subtrie.leaf_to_sample)
        ancestor_ranges = pt_batch.restoration.ancestor_segment_ranges
        if ancestor_ranges is None:
            ancestor_ranges = [[pt_batch.restoration.prefix_range] for _ in range(n)]
        temperature = orig_args.get("temperature")
        temp_is_nested = isinstance(temperature, torch.Tensor) and temperature.is_nested
        outputs: dict[str, list] = {}
        for leaf_idx, sample_idx in enumerate(pt_batch.subtrie.leaf_to_sample):
            s, e = pt_batch.restoration.segment_ranges[leaf_idx]
            ranges = list(ancestor_ranges[leaf_idx]) + [(s, e)]
            pieces = [logits_flat[a:b] for a, b in ranges if b > a]
            sample_logits = torch.cat(pieces, dim=0).unsqueeze(1)  # (len, 1, vocab)
            sample_label = pt_batch.per_sample_labels[sample_idx].unsqueeze(1)  # (len, 1)
            # logits_processor expects per-token temperature of shape (len, 1).
            # Clone: the processor does an inplace temperature[temperature <= 0] = 1e-8,
            # which must not mutate the shared nested temperature across leaves/samples.
            if temp_is_nested:
                t = temperature[sample_idx].unsqueeze(1).clone()
            elif isinstance(temperature, torch.Tensor):
                t = temperature.new_full((sample_logits.shape[0], 1), float(temperature.flatten()[0]))
            else:
                t = sample_logits.new_full(
                    (sample_logits.shape[0], 1), 1.0 if temperature is None else float(temperature)
                )
            out = logits_processor(sample_logits, label=sample_label, temperature=t, **flat_args)
            for key, val in out.items():
                if isinstance(val, torch.Tensor):
                    outputs.setdefault(key, []).append(val.reshape(-1))
        result = {}
        for key, vals in outputs.items():
            if len(vals) != n:
                continue
            result[key] = torch.nested.as_nested_tensor(vals, layout=torch.jagged)
        return result
    finally:
        clear_rope_pids(ctx.model)


# Fused-path, add boundary token to the input of Linear cross entropy, that they have 2 possible next tokens
def _prepare_lce_inputs_with_boundary(
    hidden_states: Tensor,
    labels: Tensor,
    config,
    magi_key,
    pt_batch: Optional[PrefixTreeMagiBatch],
):
    """Preprocess LCE inputs: SP gather, label pad+dispatch, boundary-pair resolution.

    Returns (hidden_ext, labels_ext, n_local, boundary_tags) where boundary_tags
    are [(boundary_position, sample_idx)] aligned with the appended tail rows.
    """
    if config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    # generate labels based on the after-packing sequence: [AB; AC] --> [ABC] where A is shared prefix
    if magi_key is not None:
        # magi ran on only this rank's CP-dispatched slice of the flat sequence:
        # reorder labels to match those rows (pad to flat length first).
        local_indices = get_position_ids(magi_key)
        flat_padded = pt_batch.tree_packed_input_ids.shape[0]
        pad = flat_padded - labels.shape[0]
        labels_full = torch.cat([labels, labels.new_zeros(pad)]) if pad > 0 else labels
        lce_labels = labels_full[local_indices]
    else:
        # flex ran on the whole flat sequence: hidden rows are already one-per-flat-position,
        # so labels only need zero-padding to the same row count.
        local_indices = None
        n_rows = hidden_states.numel() // hidden_states.shape[-1]
        pad = n_rows - labels.shape[0]
        if pad > 0:
            labels = torch.cat([labels, labels.new_zeros(pad)])
        lce_labels = labels

    hidden_2d = hidden_states.view(-1, hidden_states.shape[-1])
    registry = pt_batch.restoration.boundary_registry if (pt_batch is not None and pt_batch.restoration) else None

    # The boundary token (last token in a branch of three; multiple childs) might have n possible next token -->
    # we need to duplicate these token with multiple labels
    boundary_pairs: list[tuple[int, int]] = []
    boundary_tags: list[tuple[int, int]] = []
    if registry:
        for boundary_position, leaves in registry:
            if magi_key is not None:
                matches = (local_indices == boundary_position).nonzero()
                if matches.shape[0] == 0:
                    continue  # boundary hidden lives on another CP rank
                local_idx = int(matches[0, 0].item())
            else:
                # Non-magi: no CP dispatch — hidden rows are already in global flat
                # order, so the registry's flat position IS the row index.
                local_idx = boundary_position
            for sample_idx, next_token in leaves:
                boundary_pairs.append((local_idx, int(next_token)))
                boundary_tags.append((boundary_position, sample_idx))

    n_local = hidden_2d.shape[0]
    # Append duplicated (hidden, label) tail rows for the forks: one hidden row
    # per leaf, each scored against its OWN next token.
    if boundary_pairs:
        idx, lbl = zip(*boundary_pairs, strict=False)
        idx_t = torch.tensor(idx, device=hidden_2d.device)
        hidden_ext = torch.cat([hidden_2d, hidden_2d[idx_t]], dim=0)
        labels_ext = torch.cat(
            [lce_labels.reshape(-1), torch.tensor(lbl, device=lce_labels.device, dtype=lce_labels.dtype)]
        )
    else:
        hidden_ext, labels_ext = hidden_2d, lce_labels

    return hidden_ext, labels_ext, n_local, boundary_tags


def _run_lce_postprocess(
    logprobs_ext: Tensor,
    entropy_ext: Tensor,
    n_local: int,
    boundary_tags: list[tuple[int, int]],
    magi_key,
    pt_batch: Optional[PrefixTreeMagiBatch],
) -> tuple[Tensor, Tensor]:
    """Split boundary token added before dispatch"""
    if boundary_tags:
        logprobs = logprobs_ext[:n_local]
        entropy = entropy_ext[:n_local]
        pt_batch._boundary_local_vals = [
            (pos, sid, logprobs_ext[n_local + i]) for i, (pos, sid) in enumerate(boundary_tags)
        ]
    else:
        logprobs, entropy = logprobs_ext, entropy_ext

    # only dispatch the REAL token not the padded boundary token, tp/cp padding
    if magi_key is not None:
        logprobs = undispatch(logprobs.reshape(-1), magi_key)[: pt_batch.real_tokens]
        entropy = undispatch(entropy.reshape(-1), magi_key)[: pt_batch.real_tokens]
    return logprobs, entropy


def _run_lce(
    hidden_states: Tensor,
    output_weight: Tensor,
    labels: Tensor,
    temperature: float,
    config,
    magi_key=None,
    pt_batch: Optional[PrefixTreeMagiBatch] = None,
) -> tuple[Tensor, Tensor]:
    """Fused LCE for MAGI/flex: prepare → linear_cross_entropy → postprocess."""
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    hidden_ext, labels_ext, n_local, boundary_tags = _prepare_lce_inputs_with_boundary(
        hidden_states, labels, config, magi_key, pt_batch
    )

    logprobs_ext, entropy_ext = linear_cross_entropy(
        hidden_ext,
        output_weight,
        labels_ext,
        temperature,
        "none",
        mpu.get_tensor_model_parallel_group(),
    )

    return _run_lce_postprocess(logprobs_ext, entropy_ext, n_local, boundary_tags, magi_key, pt_batch)


def post_processing_packed_lce(
    pt_batch: PrefixTreeMagiBatch,
    magi_key=None,
) -> None:
    """Cross-CP all_gather of the boundary log-probs precomputed by the LCE
    pass (pt_batch._boundary_local_vals), stored on pt_batch._boundary_logps
    for restore_flat_to_nested."""
    registry = pt_batch.restoration.boundary_registry if pt_batch.restoration else None
    if not registry:
        return

    boundary_logps: dict[int, list[tuple[int, Tensor]]] = {}

    precomputed = getattr(pt_batch, "_boundary_local_vals", None)
    if precomputed is not None:
        local_boundary_positions = [p for p, _, _ in precomputed]
        local_sample_indices = [s for _, s, _ in precomputed]
        local_log_probs = [v for _, _, v in precomputed]
        device = local_log_probs[0].device if local_log_probs else pt_batch.tree_packed_input_ids.device
    else:
        # No boundary on this rank: participate in the collectives with zero entries.
        local_boundary_positions = []
        local_sample_indices = []
        local_log_probs = []
        device = pt_batch.tree_packed_input_ids.device

    # Cross-CP all_gather: pack (boundary_position, sample_idx, log_prob) into (max_n, 3) float32 tensor.
    cp_world = mpu.get_context_parallel_world_size()
    if cp_world > 1 and magi_key is not None:
        cp_group = mpu.get_context_parallel_group()
        local_count = len(local_log_probs)

        # Get max entry count across ranks via all_reduce(MAX) — we only need
        # the pad size, not individual per-rank counts.
        count_tensor = torch.tensor([local_count], dtype=torch.long, device=device)
        _dist.all_reduce(count_tensor, op=_dist.ReduceOp.MAX, group=cp_group)
        max_n = int(count_tensor.item())

        # Pack (boundary_position, sample_idx, log_prob) into one tensor.
        local_packed = torch.zeros(max_n, 3, dtype=torch.float32, device=device)
        local_packed[:, 1] = -1  # sentinel for padding rows
        if local_count > 0:
            local_packed[:local_count, 0] = torch.tensor(local_boundary_positions, dtype=torch.float32, device=device)
            local_packed[:local_count, 1] = torch.tensor(local_sample_indices, dtype=torch.float32, device=device)
            local_packed[:local_count, 2] = torch.stack(local_log_probs).to(torch.float32)

        # Single all_gather instead of 3 separate ones.
        all_packed = [torch.zeros_like(local_packed) for _ in range(cp_world)]
        _dist.all_gather(all_packed, local_packed, group=cp_group)

        # Validate: each (boundary_position, sample_idx) must arrive from exactly one rank.
        registry_keys = {
            (boundary_position, sample_idx) for boundary_position, leaves in registry for sample_idx, _ in leaves
        }
        key_counts: Counter[tuple[int, int]] = Counter()

        for r in range(cp_world):
            for i in range(max_n):
                sample_idx = int(all_packed[r][i, 1].item())
                if sample_idx == -1:
                    continue  # padding
                boundary_position = int(all_packed[r][i, 0].item())
                log_prob = all_packed[r][i, 2]
                key = (boundary_position, sample_idx)
                key_counts[key] += 1
                boundary_logps.setdefault(sample_idx, []).append((boundary_position, log_prob))

        # Single validation pass: unexpected, duplicate, or missing entries.
        unexpected = set(key_counts) - registry_keys
        duplicates = {k for k, v in key_counts.items() if v > 1}
        missing = registry_keys - set(key_counts)
        if unexpected or duplicates or missing:
            parts = []
            if unexpected:
                parts.append(f"{len(unexpected)} unexpected (e.g. {next(iter(unexpected))})")
            if duplicates:
                parts.append(f"{len(duplicates)} duplicate (e.g. {next(iter(duplicates))})")
            if missing:
                parts.append(f"{len(missing)} missing (e.g. {next(iter(missing))})")
            raise AssertionError(
                f"post_processing_packed_lce: registry mismatch across CP ranks — "
                f"{', '.join(parts)}. Aborting to avoid silent wrong patches."
            )
    else:
        # CP=1 (or non-magi): local triples ARE the full set, no comm/assert.
        for boundary_position, sample_idx, log_prob in zip(
            local_boundary_positions, local_sample_indices, local_log_probs, strict=True
        ):
            boundary_logps.setdefault(sample_idx, []).append((boundary_position, log_prob))
        # Fail closed: with no cross-CP gather to validate coverage, a registry
        # with boundaries must be fully covered by locally produced values.
        if registry:
            got = {(p, s) for s, lst in boundary_logps.items() for p, _ in lst}
            expected = {(p, s) for p, leaves in registry for s, _ in leaves}
            if got != expected:
                raise AssertionError(
                    f"post_processing_packed_lce: registry/produced mismatch on CP=1 path — "
                    f"missing={sorted(expected - got)[:3]}, unexpected={sorted(got - expected)[:3]}. "
                    f"Aborting to avoid silent wrong boundary patches."
                )

    # used to modify the behaviour when copy the packed tensor back to normal batch
    pt_batch._boundary_logps = boundary_logps


@dataclass
class PrefixTreeOutputProcessorContext:
    """Context passed through Megatron's output-processor hook for the prefix-tree fused path."""

    pt_batch: PrefixTreeMagiBatch
    magi_key: object
    temperature: float


def prefix_tree_output_processor(
    *,
    hidden_states,
    output_layer,
    output_weight,
    labels,
    context,
    config,
    **kwargs,
):
    """Fused prefix-tree LCE at Megatron's postprocess boundary."""
    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    # Megatron passes the shared embedding as output_weight for tied models; for
    # untied models the weight lives on output_layer.
    weight = output_weight if output_weight is not None else output_layer.weight

    logprobs, entropy = _run_lce(
        hidden_states,
        weight,
        labels,
        context.temperature,
        config,
        magi_key=context.magi_key,
        pt_batch=context.pt_batch,
    )

    # Boundary-patch: cross-CP all_gather of per-leaf boundary log-probs, before restore.
    post_processing_packed_lce(context.pt_batch, magi_key=context.magi_key)

    if has_config_logger_enabled(config):
        payload = OrderedDict(
            {
                "input_ids": kwargs.get("input_ids"),
                "position_ids": kwargs.get("position_ids"),
                "attention_mask": kwargs.get("attention_mask"),
                "decoder_input": kwargs.get("decoder_input"),
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs
    return output


def run_fused_prefix_tree(
    model,
    input_ids,
    logits_processor_args,
    labels,
    temperature,
    calculate_entropy,
    *,
    vision_model=False,
    has_vision_data=False,
):
    """The entry point"""
    if vision_model and has_vision_data:
        strip_prefix_tree_args(logits_processor_args)
        return None

    prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")

    pb = build_prefix_tree_batch(model, input_ids, logits_processor_args)
    if pb is None:
        _logger.warning("prefix_tree: build_prefix_tree_batch returned None; falling back to standard fused path")
        strip_prefix_tree_args(logits_processor_args)
        return None

    strip_prefix_tree_args(logits_processor_args)
    return _fused_core(model, pb, prefix_tree_attention, labels, temperature, calculate_entropy)


def _fused_core(model, pb, prefix_tree_attention, labels, temperature, calculate_entropy):
    """Fused-path core: forward pass with fused vocab projection (LCE)."""
    local_input_ids, local_position_ids, attn_kwargs = _prepare_attn_inputs(pb, prefix_tree_attention)
    _magi_key = attn_kwargs.pop("magi_attention_key", None)
    _flex_key = attn_kwargs.pop("flex_attention_key", None)

    m = unwrap_model(model)
    post_process = m.post_process

    # Only the last PP stage (post_process=True) needs labels for LCE.
    # Non-last stages pass labels=None; the hook runs only on post_process=True.
    real_tokens = pb.real_tokens
    if post_process:
        if pb.tree_packed_labels is None:
            _logger.warning("prefix_tree[fused]: tree_packed_labels is None; falling back to standard fused path")
            return None
        # Pass flat (deduped) labels; LCE runs on real_tokens, not total_expanded.
        labels_arg = pb.tree_packed_labels[:real_tokens]
    else:
        labels_arg = None

    # Route through stock model.forward (mcore>=0.18) with the prefix-tree
    # output_processor hook, reusing the same preprocess->decoder->postprocess
    # machinery as the non-prefix-tree path.
    set_rope_pids(m, local_position_ids)
    try:
        with prefix_tree_decoder_key_context(m, _magi_key, _flex_key):
            output_orig = m(
                input_ids=local_input_ids,
                position_ids=local_position_ids,
                attention_mask=None,
                labels=labels_arg,
                packed_seq_params=None,
                output_processor=prefix_tree_output_processor,
                output_processor_context=PrefixTreeOutputProcessorContext(
                    pt_batch=pb, magi_key=_magi_key, temperature=temperature
                ),
            )
    finally:
        clear_rope_pids(m)

    if not post_process:
        return output_orig

    # output_orig.log_probs / .entropy are (real_tokens,) flat; restore to per-sample nested.
    # log_probs: apply_boundary_patch=True to fix per-leaf boundary log-probs
    #   (pt_batch._boundary_logps was set by post_processing_packed_lce above).
    # entropy: no patch (entropy at the boundary is distribution-level, same
    #   for all leaves sharing the hidden state).
    output = {"log_probs": restore_flat_to_nested(output_orig.log_probs.reshape(-1), pb, apply_boundary_patch=True)}
    if calculate_entropy:
        output["entropy"] = restore_flat_to_nested(output_orig.entropy.reshape(-1), pb)
    return output
