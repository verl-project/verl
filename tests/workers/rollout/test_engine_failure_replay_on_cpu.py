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
"""Exercise native client replay and routing with GPU-free RPC adapters."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from ray.exceptions import ActorDiedError
from vllm.v1.engine.exceptions import EngineDeadError

from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.router import GlobalRequestLoadBalancer


class _Handle:
    """Schedule actor-like calls, including fire-and-forget release, locally."""

    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            value = getattr(self.target, name)(*args, **kwargs)
            return await value if inspect.isawaitable(value) else value

        return SimpleNamespace(remote=lambda *args, **kwargs: asyncio.create_task(call(*args, **kwargs)))


def _config():
    return OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "vllm", "response_length": 4}}})


@pytest.mark.parametrize(
    "client_class,actor_dies", [(LLMServerClient, False), (LLMServerClient, True), (FullyAsyncLLMServerClient, False)]
)
def test_pending_generation_replays_after_owner_publishes_replacement(client_class, actor_dies):
    async def main():
        calls = []

        async def failed(**kwargs):
            calls.append({**kwargs, "sampling_params": dict(kwargs["sampling_params"])})
            if actor_dies:
                raise ActorDiedError()
            return TokenOutput(
                token_ids=[7, 8],
                log_probs=[-0.1, -0.2],
                stop_reason="aborted",
                extra_fields={"engine_failed": True, "global_steps": 4},
            )

        async def healthy(**kwargs):
            calls.append({**kwargs, "sampling_params": dict(kwargs["sampling_params"])})
            tokens = [9, 10] if len(kwargs["prompt_ids"]) == 5 else [7, 8, 9, 10]
            return TokenOutput(
                token_ids=tokens,
                log_probs=[-0.3] * len(tokens),
                stop_reason="completed",
                extra_fields={"global_steps": 5 if client_class is FullyAsyncLLMServerClient else 4},
            )

        old, new = _Handle(SimpleNamespace(generate=failed)), _Handle(SimpleNamespace(generate=healthy))
        router = GlobalRequestLoadBalancer({"same-address": old})
        client = client_class(_config(), _Handle(router), engine_recovery_timeout=5)
        params = {"max_tokens": 4, "logprobs": True}

        async def owner():
            while router.get_all_servers():
                await asyncio.sleep(0)
            router.add_servers({"same-address": new})
            # A late failure or release must leave this new actor untouched.
            assert not router.remove_server_if_current("same-address", old)
            router.acquire_server("other-session")
            router.release_server("same-address", expected_handle=old)
            assert router.get_inflight_count("same-address") == 1
            router.release_server("same-address", expected_handle=new)

        task = asyncio.create_task(owner())
        result = await client.generate("session", prompt_ids=[1, 2, 3], sampling_params=params)
        await task
        await asyncio.sleep(0)
        assert len(calls) == 2 and result.token_ids == [7, 8, 9, 10]
        assert len(result.log_probs) == 4
        assert result.stop_reason == ("length" if client_class is FullyAsyncLLMServerClient else "completed")
        assert calls[0]["prompt_ids"] == [1, 2, 3]
        if client_class is FullyAsyncLLMServerClient:
            assert calls[1]["prompt_ids"] == [1, 2, 3, 7, 8]
            assert calls[1]["sampling_params"]["max_tokens"] == 2
            assert result.log_probs[:2] == [-0.1, -0.2]
            assert result.extra_fields["min_global_steps"] == 4 and result.extra_fields["max_global_steps"] == 5
        else:
            assert calls[1]["prompt_ids"] == [1, 2, 3]
            assert calls[1]["sampling_params"]["max_tokens"] == 4
        assert params == {"max_tokens": 4, "logprobs": True}
        assert router.get_inflight_count("same-address") == 0

    asyncio.run(main())


@pytest.mark.parametrize("error,recovery_timeout", [(ValueError("invalid request"), 5), (EngineDeadError(), None)])
def test_errors_outside_enabled_engine_replay_propagate(error, recovery_timeout):
    async def main():
        async def failed(**kwargs):
            raise error

        server = _Handle(SimpleNamespace(generate=failed))
        router = GlobalRequestLoadBalancer({"server": server})
        client = LLMServerClient(_config(), _Handle(router), engine_recovery_timeout=recovery_timeout)
        with pytest.raises(type(error)):
            await client.generate("session", prompt_ids=[1], sampling_params={"max_tokens": 4})
        await asyncio.sleep(0)
        assert router.get_all_servers() == ["server"] and router.get_inflight_count("server") == 0

    asyncio.run(main())
