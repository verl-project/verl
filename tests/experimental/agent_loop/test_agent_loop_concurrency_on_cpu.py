# Copyright 2026 Tencent Ltd. and/or its affiliates
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

from types import SimpleNamespace

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker


def test_eval_concurrency_uses_eval_cap_before_rollout_cap():
    config = SimpleNamespace(
        async_training={
            "max_concurrent_rollouts": 192,
            "max_concurrent_eval_rollouts": 64,
        }
    )

    assert AgentLoopWorker._resolve_max_concurrent_agent_loops(config, batch_size=300, validate=True) == 64
    assert AgentLoopWorker._resolve_max_concurrent_agent_loops(config, batch_size=300, validate=False) == 192


def test_eval_concurrency_falls_back_to_rollout_cap():
    config = SimpleNamespace(async_training={"max_concurrent_rollouts": 96})

    assert AgentLoopWorker._resolve_max_concurrent_agent_loops(config, batch_size=300, validate=True) == 96


def test_concurrency_never_exceeds_batch_size():
    config = SimpleNamespace(
        async_training={
            "max_concurrent_rollouts": 192,
            "max_concurrent_eval_rollouts": 64,
        }
    )

    assert AgentLoopWorker._resolve_max_concurrent_agent_loops(config, batch_size=12, validate=True) == 12
