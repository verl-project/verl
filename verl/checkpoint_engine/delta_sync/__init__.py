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
"""Delta weight sync primitives for the checkpoint engine.

This package contains the framework-agnostic core of "send only the elements
that changed" weight sync: the pinned-CPU snapshot, bytewise diff, encode, and
bucket logic. :class:`~verl.checkpoint_engine.delta_checkpoint_engine.DeltaCheckpointEngine`
consumes these flushes and broadcasts them over NCCL; the rollout worker decodes
them back into full tensors locally.

Design follows THUDM/slime's delta-sync implementation
(``slime/backends/megatron_utils/update_weight/update_weight_from_distributed_delta.py``).
"""

from .delta_state import DeltaState, ParamDiff
from .encode import (
    DeltaBucket,
    DeltaEncodingName,
    EncodedChunk,
    bytewise_diff_mask,
    checksum,
    encode_chunk,
)
from .wrapper import DeltaFlush, iter_delta_flushes

__all__ = [
    "DeltaBucket",
    "DeltaEncodingName",
    "DeltaFlush",
    "DeltaState",
    "EncodedChunk",
    "ParamDiff",
    "bytewise_diff_mask",
    "checksum",
    "encode_chunk",
    "iter_delta_flushes",
]
