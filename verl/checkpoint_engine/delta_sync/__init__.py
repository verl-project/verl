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

This package holds the framework-agnostic wire schema of "send only the
elements that changed" weight sync: the per-flush parameter spec
(:class:`DeltaParam` / :class:`DeltaFlush`) and the payload checksum. The
sharded engines assemble these flushes from backend-provided shard exports and
broadcast them over NCCL; the rollout backend decodes them in place.

Design follows THUDM/slime's delta-sync implementation
(``slime/backends/megatron_utils/update_weight/update_weight_from_distributed_delta.py``).
"""

from .encode import DeltaFlush, DeltaParam, checksum

__all__ = [
    "DeltaFlush",
    "DeltaParam",
    "checksum",
]
