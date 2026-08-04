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

import asyncio

from verl.workers.reward_manager import prime


async def _fake_parallel_compute_score_async(*args, **kwargs):
    return [0.5]


def test_run_reward_scoring_restores_previous_event_loop(monkeypatch):
    monkeypatch.setattr(prime, "parallel_compute_score_async", _fake_parallel_compute_score_async)
    previous_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(previous_loop)

    try:
        scores = prime.run_reward_scoring(
            evaluation_func=None,
            completions=["completion"],
            references=["reference"],
            tasks=["task"],
            num_processes=1,
        )

        assert scores == [0.5]
        assert asyncio.get_event_loop() is previous_loop
        assert not previous_loop.is_closed()
    finally:
        asyncio.set_event_loop(None)
        previous_loop.close()
