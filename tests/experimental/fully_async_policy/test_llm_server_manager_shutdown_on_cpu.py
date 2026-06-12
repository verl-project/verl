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

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager


class _ShutdownReplica:
    def __init__(self):
        self.shutdown_calls = 0

    async def shutdown(self):
        self.shutdown_calls += 1


def test_fully_async_llm_server_manager_shutdown_includes_hybrid_replicas():
    manager = object.__new__(FullyAsyncLLMServerManager)
    standalone_replica = _ShutdownReplica()
    sleeping_hybrid_replica = _ShutdownReplica()
    active_hybrid_replica = _ShutdownReplica()
    manager.rollout_replicas = [standalone_replica, active_hybrid_replica]
    manager.hybrid_replicas = {
        "hybrid_sleeping": sleeping_hybrid_replica,
        "hybrid_active": active_hybrid_replica,
    }

    manager.shutdown()

    assert standalone_replica.shutdown_calls == 1
    assert sleeping_hybrid_replica.shutdown_calls == 1
    assert active_hybrid_replica.shutdown_calls == 1
