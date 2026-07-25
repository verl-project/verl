# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Shared routing value types consumed by strategies and the Balancer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplicaInfo:
    """Descriptor of a routable replica.

    Carries only the replica id; the actor handle stays in the Balancer's pool.
    """

    replica_id: str
