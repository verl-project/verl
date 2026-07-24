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
"""Segment-based grouping for prefix-tree optimization.

Provides a generic interface for building prefix trees from pre-computed
segment metadata, avoiding expensive token-by-token detection when the
grouping structure is known at data creation time.

Segment Format:
    Each sample is described as a list of segments: [(hash, length), ...]
    - hash: Any hashable value (int, str, UUID) identifying the segment content
    - length: Number of tokens in the segment

Tree Building:
    Samples sharing the same hash for segment i share that prefix path.
    The tree is built by matching segment hashes level by level.

Examples:
    GRPO (single prompt, n rollouts):
        Sample 0: [("uuid-p0", 100), ("resp-hash-0", 50)]
        Sample 1: [("uuid-p0", 100), ("resp-hash-1", 45)]
        -> Share first segment (uuid-p0), diverge at second
"""

from typing import Hashable

import numpy as np


def _default_uid_hash(uid: str) -> int:
    return hash(uid) & 0xFFFFFFFF


def create_segment_metadata(
    segments: list[list[tuple[Hashable, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Create segment metadata arrays for prefix-tree building.

    Returns:
        (segment_hashes, segment_lengths) numpy arrays with object dtype.
        Supports fancy indexing during reorder.
    """
    segment_hashes = []
    segment_lengths = []
    for sample_segments in segments:
        hashes, lengths = [], []
        for hash_val, length in sample_segments:
            if isinstance(hash_val, str):
                hash_val = _default_uid_hash(hash_val)
            elif not isinstance(hash_val, int):
                hash_val = hash(hash_val) & 0xFFFFFFFF
            hashes.append(hash_val)
            lengths.append(length)
        segment_hashes.append(hashes)
        segment_lengths.append(lengths)
    return (
        np.array(segment_hashes, dtype=object),
        np.array(segment_lengths, dtype=object),
    )


def create_grpo_segment_metadata(
    prompt_uids: list[str],
    prompt_lengths: list[int],
    rollout_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create segment metadata for GRPO: one shared-prompt segment per sample.

    Args:
        prompt_uids: Per-sample UUID. Same UUID = same prompt prefix.
        prompt_lengths: Prompt token length for each sample.
        rollout_n: Number of rollouts per prompt (for validation only).

    Returns:
        (segment_hashes, segment_lengths) arrays.
    """
    batch_size = len(prompt_uids)
    if batch_size % rollout_n != 0:
        raise ValueError(f"batch_size {batch_size} not divisible by rollout_n {rollout_n}")
    segments = [[(uid, p_len)] for uid, p_len in zip(prompt_uids, prompt_lengths, strict=False)]
    return create_segment_metadata(segments)


def group_by_segment_hash(
    segment_hashes: np.ndarray,
    segment_lengths: np.ndarray,
    level: int = 0,
) -> dict[int, list[tuple[int, int]]]:
    """Group samples by their segment hash at a given level.

    Returns:
        Dict mapping hash → list of (sample_idx, segment_length).
    """
    groups: dict[int, list[tuple[int, int]]] = {}
    for sample_idx in range(len(segment_hashes)):
        sample_hashes = segment_hashes[sample_idx]
        sample_lengths = segment_lengths[sample_idx]
        if level >= len(sample_hashes):
            continue
        hash_val = int(sample_hashes[level])
        length = int(sample_lengths[level])
        groups.setdefault(hash_val, []).append((sample_idx, length))
    return groups
