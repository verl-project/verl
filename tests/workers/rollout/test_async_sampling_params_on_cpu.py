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
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class _RolloutConfig:
    max_model_len = 100
    prompt_length = 16
    response_length = 8
    enable_rollout_routing_replay = False

    def get(self, key, default=None):
        return getattr(self, key, default)


class _ModelConfig:
    processor = None
    lora_rank = 0
    lora = {}


class _LogProb:
    def __init__(self, value: float):
        self.logprob = value


class _FakeVllmEngine:
    def __init__(self):
        self.calls = []
        self.call_count = 0

    def generate(self, **kwargs):
        sampling_params = kwargs["sampling_params"]
        self.calls.append(sampling_params)
        self.call_count += 1

        token_ids = [10 * self.call_count + 1, 10 * self.call_count + 2]
        logprobs = None
        if sampling_params.logprobs is not None:
            logprobs = [
                {token_ids[0]: _LogProb(-0.1 * self.call_count)},
                {token_ids[1]: _LogProb(-0.2 * self.call_count)},
            ]
        output = SimpleNamespace(token_ids=token_ids, logprobs=logprobs, finish_reason="stop")
        request_output = SimpleNamespace(outputs=[output])

        async def _generator():
            yield request_output

        return _generator()

    async def list_loras(self):
        return []


def _make_vllm_http_server():
    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.config = _RolloutConfig()
    server.model_config = _ModelConfig()
    server.engine = _FakeVllmEngine()
    server.global_steps = 0
    return server


def test_vllm_generate_keeps_sampling_params_reusable_across_turns():
    async def _run():
        server = _make_vllm_http_server()
        sampling_params = {"max_tokens": 2, "logprobs": True, "temperature": 0.0}

        first = await vLLMHttpServer.generate(server, [1, 2, 3], sampling_params, "turn-1")
        second = await vLLMHttpServer.generate(server, [1, 2, 3, 11, 12], sampling_params, "turn-2")

        assert first.log_probs == [-0.1, -0.2]
        assert second.log_probs == [-0.2, -0.4]
        assert sampling_params == {"max_tokens": 2, "logprobs": True, "temperature": 0.0}
        assert [call.max_tokens for call in server.engine.calls] == [2, 2]
        assert [call.logprobs for call in server.engine.calls] == [0, 0]

    asyncio.run(_run())
