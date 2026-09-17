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
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


@pytest.mark.parametrize("cache_salt", [None, "cache-namespace"])
def test_generate_passes_cache_namespace_with_multimodal_prompt(cache_salt: str | None) -> None:
    server = object.__new__(vLLMHttpServer)
    server._disaggregation_role = None
    server.config = OmegaConf.create({"max_model_len": 16, "full_determinism": False})
    server.model_config = SimpleNamespace(processor=None, lora_rank=0, lora={})
    server._submission_paused = False
    server._admitting = 0
    server.replica_rank = 0
    server.global_steps = 0

    async def generate(**kwargs):
        yield SimpleNamespace(outputs=[])

    server.engine = SimpleNamespace(generate=Mock(side_effect=generate))
    output = asyncio.run(
        server.generate(
            prompt_ids=[1, 2, 3],
            sampling_params={"max_tokens": 4},
            request_id="request-1",
            image_data=["image"],
            cache_salt=cache_salt,
        )
    )

    prompt = server.engine.generate.call_args.kwargs["prompt"]
    assert prompt["prompt_token_ids"] == [1, 2, 3]
    assert prompt["multi_modal_data"] == {"image": ["image"]}
    if cache_salt is None:
        assert "cache_salt" not in prompt
    else:
        assert prompt["cache_salt"] == cache_salt
    assert output.stop_reason == "aborted"
    assert server._admitting == 0


@pytest.mark.parametrize("reset_connector", [True, False])
def test_clear_kv_cache_can_preserve_shared_connector(reset_connector):
    server = object.__new__(vLLMHttpServer)
    server.node_rank = 0
    server.engine = SimpleNamespace(
        reset_prefix_cache=AsyncMock(return_value=True),
        reset_mm_cache=AsyncMock(),
        reset_encoder_cache=AsyncMock(),
    )

    kwargs = {} if reset_connector else {"reset_connector": False}
    asyncio.run(server.clear_kv_cache(**kwargs))

    server.engine.reset_prefix_cache.assert_awaited_once_with(reset_connector=reset_connector)
    server.engine.reset_mm_cache.assert_awaited_once()
    server.engine.reset_encoder_cache.assert_awaited_once()


def test_clear_kv_cache_reports_failed_reset():
    server = object.__new__(vLLMHttpServer)
    server.node_rank = 0
    server.engine = SimpleNamespace(reset_prefix_cache=AsyncMock(return_value=False))

    with pytest.raises(RuntimeError, match="prefix-cache reset failed"):
        asyncio.run(server.clear_kv_cache(reset_connector=False))
