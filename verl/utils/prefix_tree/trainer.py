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

"""Prefix-tree helpers consumed by verl trainers (SFT, PPO)."""

from __future__ import annotations

import time

import numpy as np
import torch

from verl.utils.prefix_tree.dynamic import compute_prefix_tree_metrics, greedy_build_tries
from verl.utils.prefix_tree.tree import _is_prefix_tree_enabled


def pt_metrics(
    metrics: dict,
    input_ids,
    config_or_data: dict,
    attention_mask=None,
    max_token_len_per_gpu: int | None = None,
    trie=None,
    leaf_idx=None,
) -> None:
    """Compute prefix_tree/* metrics if use_prefix_tree enabled (no-op otherwise)."""
    if not _is_prefix_tree_enabled(config_or_data):
        return
    metrics.update(
        compute_prefix_tree_metrics(
            input_ids,
            attention_mask=attention_mask,
            max_token_len_per_gpu=max_token_len_per_gpu,
            trie=trie,
            leaf_idx=leaf_idx,
        )
    )


def build_global_trie(input_ids, attention_mask=None, *, metrics=None):
    """Build global prefix trie via greedy token-by-token detection.

    Returns (trie, leaf_idx, build_time). trie is None when there's no sharing.
    Callers attach trie + leaf_idx as they need (DataProto, TQ, etc.)."""
    if attention_mask is not None:
        seqs = [ids[mask.bool()].tolist() or [0] for ids, mask in zip(input_ids, attention_mask, strict=False)]
    else:
        seqs = [ids.tolist() for ids in input_ids]

    _t0 = time.perf_counter()
    trie, _ = greedy_build_tries(seqs)
    _t1 = time.perf_counter()
    if metrics is not None:
        metrics["actor/prefix_tree/tree_build_time_s"] = _t1 - _t0
    if trie is None:
        return None, None, _t1 - _t0

    # leaf_idx[seq_id] = deepest node whose sequence_ids include it (DFS node_idx
    # increases with depth, so the max is the sequence's end node). Handles strict
    # prefixes, which end at an internal node.
    leaf_idx = np.full(len(seqs), -1, dtype=np.int64)
    for node_idx, node in enumerate(trie.nodes):
        for seq_id in node.sequence_ids:
            if node_idx > leaf_idx[seq_id]:
                leaf_idx[seq_id] = node_idx
    if (leaf_idx < 0).any():
        missing = np.where(leaf_idx < 0)[0].tolist()
        raise ValueError(
            f"build_global_trie: {len(missing)} samples have no leaf assigned "
            f"(first {missing[:8]}). The trie did not cover every sequence."
        )
    return trie, torch.from_numpy(leaf_idx), _t1 - _t0
