# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Context manager that temporarily switches the global Ulysses SP group.

DeepSpeed workers may colocate multiple roles (actor/ref/critic) inside a single
process, but the Ulysses kernels rely on a module-level singleton for the
sequence-parallel process group.  This manager allows each role to stash its own
group and restore it around forward/backward regions to avoid cross-role
interference.
"""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist

from verl.utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group


class DeepSpeedUlyssesShardingManager:
    """Simple scoped switch for the global Ulysses process group."""

    def __init__(self, process_group: Optional[dist.ProcessGroup]):
        self.process_group = process_group
        self._prev_group: Optional[dist.ProcessGroup] = None

    def __enter__(self):
        if self.process_group is None:
            return self

        self._prev_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.process_group)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.process_group is None:
            return

        set_ulysses_sequence_parallel_group(self._prev_group)
        self._prev_group = None
