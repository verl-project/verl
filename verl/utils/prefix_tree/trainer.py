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

"""Prefix-tree helpers consumed by verl trainers (SFT, PPO).

Every public function here is a single call that checks *config* internally;
the caller never needs to gate on ``use_prefix_tree``.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from verl.utils.prefix_tree.dynamic import compute_prefix_tree_metrics, greedy_build_tries
from verl.utils.prefix_tree.segment_grouper import create_grpo_segment_metadata
from verl.utils.prefix_tree.tree import _is_prefix_tree_enabled, build_global_tree_from_segments


def apply_engine_config(engine_config, config_or_data: dict) -> None:
    """Thread prefix-tree flags from config into *engine_config*."""
    engine_config.use_prefix_tree = config_or_data.get("use_prefix_tree", False)
    engine_config.prefix_tree_attention = config_or_data.get("prefix_tree_attention", "magi")


def add_meta_info(meta_dict: dict, config_or_data: dict) -> None:
    """Add prefix-tree entries to a meta-info dict (mutates in-place)."""
    meta_dict["use_prefix_tree"] = config_or_data.get("use_prefix_tree", False)
    meta_dict["prefix_tree_attention"] = config_or_data.get("prefix_tree_attention", "magi")


def pt_metrics(
    metrics: dict,
    input_ids,  # TODO: use PrefixTrie / PrefixSubTrie
    config_or_data: dict,
    attention_mask=None,
    max_token_len_per_gpu: int | None = None,
    micro_batch_size: int = 0,
    trie=None,
    leaf_idx=None,
) -> None:
    """Compute prefix-sharing metrics if *use_prefix_tree* is enabled.

    Updates *metrics* in-place with the trie-structure-invariant metrics
    ``prefix_tree/global_shared_ratio``, ``prefix_tree/packed_tokens`` and
    ``prefix_tree/raw_tokens``.  Pass *attention_mask* to strip padding from
    2-D padded tensors.  Pass *trie* to skip the internal
    ``greedy_build_tries`` when the caller already built one (e.g. attached
    to ``batch.meta_info["prefix_tree"]``).

    Note: ``micro_batch_shared_ratio`` is NOT computed here.  The previous
    pre-forward computation (on the full batch, before DP dispatch and
    reorder) did not match the actual micro-batches the engine dispatches.
    The accurate version is computed inside
    ``prepare_prefix_tree_micro_batches`` (from the actual ``batch_idx_list``
    grouping) and surfaced through the engine output via
    ``maybe_collect_mbs_metric``; the PPO trainer threads it into the
    ``actor/prefix_tree/micro_batch_shared_ratio`` metric from the OLP path.

    The ``max_token_len_per_gpu``, ``micro_batch_size`` and ``leaf_idx``
    parameters are kept for backward-compat callers but are no longer used.
    """
    if not _is_prefix_tree_enabled(config_or_data):
        return
    metrics.update(
        compute_prefix_tree_metrics(
            input_ids,
            attention_mask=attention_mask,
            max_token_len_per_gpu=max_token_len_per_gpu,
            micro_batch_size=micro_batch_size,
            trie=trie,
            leaf_idx=leaf_idx,
        )
    )


def attach_segment_metadata(batch, rollout_n: int) -> None:
    """Attach segment metadata for prefix-tree fast path (GRPO).

    Creates segment_hashes and segment_lengths from the batch's prompt UIDs and
    prompt lengths, storing them in non_tensor_batch as numpy object arrays so
    they survive reorder()/chunk()/to_tensordict() round-trips.
    """
    if rollout_n < 2:
        return
    prompt_uids = batch.non_tensor_batch.get("uid", None)
    if prompt_uids is None:
        return
    attention_mask = batch.batch["attention_mask"]
    response_length = batch.batch["responses"].size(1)
    prompt_lengths = attention_mask[:, :-response_length].sum(dim=-1).cpu().tolist()

    segment_hashes, segment_lengths = create_grpo_segment_metadata(
        prompt_uids=list(prompt_uids),
        prompt_lengths=prompt_lengths,
        rollout_n=rollout_n,
    )
    batch.non_tensor_batch["segment_hashes"] = segment_hashes
    batch.non_tensor_batch["segment_lengths"] = segment_lengths


def build_global_trie(batch, *, metrics=None, rollout_n=None) -> float:
    """Build global prefix trie from segment metadata (or token-by-token fallback)
    and attach to batch. Mutates batch in-place.

    - trie -> batch.meta_info["prefix_tree"] (TrieNode root, shared across samples)
    - leaf_idx -> batch.batch["leaf_idx"] (torch.long tensor, sample -> leaf flat_idx)

    Both survive DataProto.reorder/chunk/slice/concat/repeat natively:
    batch tensors propagate via torch indexing; meta_info wraps as NonTensorData.

    Args:
        batch: DataProto to mutate.
        metrics: Optional metrics dict. When provided, sets
            ``metrics["prefix_tree/timing_s"]`` and
            ``batch.meta_info["prefix_tree_path_tag"]`` after the build.
        rollout_n: Optional rollout.n. When provided, calls
            :func:`attach_segment_metadata` first (no-op if rollout_n < 2 or
            no uids).

    Returns:
        Wall-clock seconds spent building the trie (segment fast path or greedy
        fallback), excluding input prep and leaf_idx assignment.
    """
    if rollout_n is not None:
        attach_segment_metadata(batch, rollout_n)
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch.get("attention_mask", None)
    if attention_mask is not None:
        seqs = [input_ids[i][attention_mask[i].bool()].tolist() or [0] for i in range(len(input_ids))]
    else:
        seqs = [input_ids[i].tolist() for i in range(len(input_ids))]
    total_raw = sum(len(s) for s in seqs)

    seg_hashes = batch.non_tensor_batch.get("segment_hashes", None)
    seg_lengths = batch.non_tensor_batch.get("segment_lengths", None)
    _t0 = time.perf_counter()
    trie = None
    if seg_hashes is not None and seg_lengths is not None:
        trie = build_global_tree_from_segments(seqs, seg_hashes, seg_lengths)
    if trie is None:
        tries, _ = greedy_build_tries(seqs, max_tokens_per_tree=total_raw * 10)
        if tries and total_raw > 0:
            trie = tries[0]
    _t1 = time.perf_counter()
    if metrics is not None:
        metrics["prefix_tree/timing_s"] = _t1 - _t0
        batch.meta_info["prefix_tree_path_tag"] = "segment" if seg_hashes is not None else "uniform"
    if trie is None:
        return 0.0

    leaf_idx = np.full(len(seqs), -1, dtype=np.int64)
    for flat_idx, node in enumerate(trie.nodes):
        if not node.children:  # leaf
            for seq_id in node.sequence_ids:
                leaf_idx[seq_id] = flat_idx

    batch.meta_info["prefix_tree"] = trie
    batch.batch["leaf_idx"] = torch.from_numpy(leaf_idx)
    return _t1 - _t0
