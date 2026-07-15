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
"""The shard-export contract between training engines and the sharded delta engine.

``BaseEngine.get_per_tensor_param_shard`` yields ``(name, local_shard, ShardSpec)``
per local parameter, in an order identical on every rank. All layout knowledge
lives in the spec, provided by the trainer backend, so the checkpoint engine
stays backend-agnostic. Two profiles:

**Closed-form (fast path)** -- the shard maps to ONE HF tensor and shard-local flat
positions have a closed-form mapping to full-tensor flat positions (e.g. FSDP
``Shard(0)``): set ``translate`` / ``hf_name`` / ``full_shape`` / ``full_numel`` /
``dense_offset``. Sparse deltas are translated rank-locally and gathered without
per-rank boundaries; the dense first sync gathers by flat offset.

**Rebuild (general path)** -- the shard participates in an arbitrary pure-permutation
conversion, possibly spanning several HF tensors (e.g. Megatron fused qkv): set
``gather_group`` + ``rebuild_dense`` and leave ``translate`` None. Sparse deltas are
gathered per rank within ``gather_group``; rank 0 rebuilds NaN-sentinel shards, runs
``rebuild_dense``, and reads the non-NaN entries as the HF-coordinate delta (valid
because a pure permutation preserves sentinel positions). The dense first sync runs
``rebuild_dense`` on the real shards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

__all__ = ["ShardSpec"]


@dataclass
class ShardSpec:
    """Placement descriptor for one exported local parameter shard."""

    # Whether this rank's copy should be counted (False for redundant replicas).
    contributes: bool
    # Shape of this rank's local shard (uniform across the gather group).
    shard_shape: tuple

    # ---- closed-form profile ----
    # Map shard-local flat positions to full-tensor flat positions (rank-local).
    translate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    hf_name: Optional[str] = None
    full_shape: Optional[tuple] = None
    full_numel: Optional[int] = None
    # Flat offset of this rank's shard in the full parameter (dense first sync).
    dense_offset: Optional[int] = None

    # ---- rebuild profile ----
    # Collective group whose members each hold one shard (None = default group).
    gather_group: Optional[object] = None
    # Pure-permutation mapping from the group's dense shards to HF tensors.
    rebuild_dense: Optional[Callable[[list[torch.Tensor]], list[tuple[str, torch.Tensor]]]] = None
