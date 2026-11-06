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
"""The rollout sessions of a prompt share its media preprocessing."""

import asyncio

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.preprocess_cache import PromptPreprocessCache
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.ray_utils import get_event_loop


@pytest.mark.asyncio
async def test_concurrent_lookups_of_same_inputs_compute_once():
    cache = PromptPreprocessCache()
    calls = []

    async def compute():
        calls.append(None)
        await asyncio.sleep(0.01)
        return len(calls)

    image = object()
    messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    results = await asyncio.gather(*(cache.get_or_compute("step", list(messages), compute) for _ in range(4)))
    assert results == [1, 1, 1, 1]

    # Equal-valued but distinct media objects are other inputs.
    other = [{"role": "user", "content": [{"type": "image", "image": object()}]}]
    assert await cache.get_or_compute("step", other, compute) == 2
    assert await cache.get_or_compute("other_step", list(messages), compute) == 3


@pytest.mark.asyncio
async def test_cache_key_is_not_changed_by_callers_appending_to_inputs():
    cache = PromptPreprocessCache()
    messages = [{"role": "user", "content": "q"}]

    async def compute():
        return "value"

    await cache.get_or_compute("step", messages, compute)
    messages.append({"role": "tool", "content": "a"})

    async def fail():
        raise AssertionError("cache missed")

    assert await cache.get_or_compute("step", [{"role": "user", "content": "q"}], fail) == "value"


@pytest.mark.asyncio
async def test_waiters_retry_when_the_first_compute_fails():
    cache = PromptPreprocessCache()
    started = asyncio.Event()

    async def fail():
        started.set()
        await asyncio.sleep(0.01)
        raise RuntimeError("decode failed")

    async def compute():
        return "value"

    first = asyncio.create_task(cache.get_or_compute("step", ["q"], fail))
    await started.wait()
    waiter = asyncio.create_task(cache.get_or_compute("step", ["q"], compute))
    with pytest.raises(RuntimeError):
        await first
    assert await waiter == "value"


class _CountingDataset:
    calls = 0

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        cls.calls += 1
        return [object()], None


class _CountingBuilder:
    def __init__(self):
        self.calls = 0

    def build_initial_tokens(self, messages, tools=None, images=None, videos=None, audios=None, mm_inputs_out=None):
        self.calls += 1
        mm_inputs_out["pixel_values"] = torch.zeros(len(images), 4)
        return [1, 2, 3]


def _agent_loop(cache: PromptPreprocessCache, builder: _CountingBuilder) -> SingleTurnAgentLoop:
    agent_loop = object.__new__(SingleTurnAgentLoop)
    agent_loop.processor = object()
    agent_loop.dataset_cls = _CountingDataset
    agent_loop.data_config = OmegaConf.create({})
    agent_loop.rollout_config = OmegaConf.create({"prompt_length": 16})
    agent_loop.continuous_token_builder = builder
    agent_loop.loop = get_event_loop()
    agent_loop.preprocess_cache = cache
    return agent_loop


@pytest.mark.asyncio
async def test_sessions_of_a_prompt_decode_media_and_render_prompt_once():
    _CountingDataset.calls = 0
    cache, builder = PromptPreprocessCache(), _CountingBuilder()
    raw_prompt = [{"role": "user", "content": [{"type": "image", "image": "a.png"}, {"type": "text", "text": "q"}]}]

    async def session():
        agent_loop = _agent_loop(cache, builder)
        messages = list(raw_prompt)
        multi_modal_data = await agent_loop.process_multi_modal_info(messages)
        mm_inputs = {}
        prompt_ids = await agent_loop.ct_build_initial_tokens(
            messages, images=multi_modal_data["images"], mm_inputs_out=mm_inputs
        )
        return multi_modal_data, prompt_ids, mm_inputs

    sessions = await asyncio.gather(*(session() for _ in range(4)))
    assert _CountingDataset.calls == 1 and builder.calls == 1

    (first_data, first_ids, first_mm), (second_data, second_ids, second_mm) = sessions[:2]
    assert first_data["images"][0] is second_data["images"][0]
    assert first_mm["pixel_values"] is second_mm["pixel_values"]
    assert first_ids == second_ids == [1, 2, 3]
    # A session appending tool images or editing its prompt ids does not affect its siblings.
    first_data["images"].append(object())
    first_ids.append(4)
    assert len(second_data["images"]) == 1 and second_ids == [1, 2, 3]


@pytest.mark.asyncio
async def test_agent_loop_without_cache_preprocesses_every_session():
    _CountingDataset.calls = 0
    builder = _CountingBuilder()
    for _ in range(2):
        agent_loop = _agent_loop(None, builder)
        multi_modal_data = await agent_loop.process_multi_modal_info([{"role": "user", "content": "q"}])
        await agent_loop.ct_build_initial_tokens([], images=multi_modal_data["images"], mm_inputs_out={})
    assert _CountingDataset.calls == 2 and builder.calls == 2
