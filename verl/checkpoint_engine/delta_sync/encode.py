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
"""Position+value encoding and bucketing for delta sync.

Both encodings share one on-wire layout (a uint8 positions blob plus a
parameter-dtype values tensor with a per-parameter manifest); decoders
dispatch on metadata.

* ``indices`` -- int32 absolute positions (4 bytes / nnz)
* ``deltas``  -- uint16 gap-deltas with uint32 per-parameter fallback

Values are sent verbatim in the parameter's dtype regardless of encoding.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import torch


DeltaEncodingName = Literal["indices", "deltas"]


# ---------- diff ----------------------------------------------------------



@dataclass
class DeltaParam:
    """Per-parameter manifest entry for a single chunk / bucket.

    Offsets are byte offsets into the surrounding ``__positions__`` blob and
    element offsets into the surrounding ``__values__`` tensor.
    """

    name: str
    dtype: str
    shape: list[int]
    pos_start: int
    pos_end: int
    pos_width: int  # 2 or 4
    val_start: int
    val_end: int


def checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    """Wire-corruption check; sender computes pre-flush, receiver post-recv.

    Uses ``torch.hash_tensor`` (XOR-reduce over uint64 bitcast); one reduction
    plus one ``.item()`` sync per argument.
    """
    p = int(torch.hash_tensor(positions).item()) if positions.numel() else 0
    v = int(torch.hash_tensor(values).item()) if values.numel() else 0
    return p ^ (v << 1)


# ---------- encode --------------------------------------------------------
