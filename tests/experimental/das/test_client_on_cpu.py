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
"""``wrap_llm_client`` over the real ``LLMServerClient`` / ``FullyAsyncLLMServerClient`` with fake servers."""

import re

import pytest
from omegaconf import OmegaConf

from tests.experimental.das.fakes import FakeActor, FakeServer
from verl.experimental.das.client import DAS_PUSH_ENABLED, DASClientMixin, wrap_llm_client
from verl.experimental.das.problem_id import hash_problem_id
from verl.experimental.das.service import SuffixDataService
from verl.workers.config.das import DASConfig
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.router import GlobalRequestLoadBalancer

DASP_RE = re.compile(r"^dasp[0-9a-f]{16}-[0-9a-f]{32}$")
PROMPT = [11, 12, 13, 14]


def make_config(full_determinism=False):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"name": "vllm", "full_determinism": full_determinism, "response_length": 32}
            }
        }
    )


def make_client(cls=LLMServerClient, *, server=None, full_determinism=False, track_weight_version=False):
    server = server or FakeServer()
    balancer = FakeActor(GlobalRequestLoadBalancer({"s0": FakeActor(server)}))
    client = cls(make_config(full_determinism), load_balancer_handle=balancer)
    cfg = DASConfig(enable=True, push_batch_size=1)
    wrapped = wrap_llm_client(client, cfg, track_weight_version=track_weight_version)
    store = SuffixDataService(cfg)
    wrapped._das_service = FakeActor(store)  # what ray.get_actor would return
    return wrapped, server, store


def pushes(store):
    return store.get_metrics()["das_pushes_received"]


@pytest.mark.asyncio
async def test_engine_request_ids_carry_problem_hash_and_siblings_share_it():
    client, server, store = make_client()
    out_a = await client.generate("traj-a", prompt_ids=PROMPT, sampling_params={})
    await client.generate("traj-b", prompt_ids=PROMPT, sampling_params={})  # GRPO sibling
    assert out_a.token_ids == [7, 8, 9]
    ids = [c["request_id"] for c in server.calls]
    assert all(DASP_RE.match(i) for i in ids)
    assert ids[0] != ids[1]
    assert ids[0][:20] == ids[1][:20] == f"dasp{hash_problem_id(PROMPT)}"
    # verl's sticky session id is still forwarded unchanged.
    assert [c["session_id"] for c in server.calls] == ["traj-a", "traj-b"]
    assert pushes(store) == 2
    assert isinstance(client, DASClientMixin)
    assert isinstance(client, LLMServerClient)


@pytest.mark.asyncio
async def test_second_turn_with_grown_prompt_keeps_first_turn_hash():
    client, server, _ = make_client()
    await client.generate("traj", prompt_ids=PROMPT, sampling_params={})
    await client.generate("traj", prompt_ids=PROMPT + [7, 8, 9, 50, 51], sampling_params={})
    first, second = (c["request_id"] for c in server.calls)
    assert first[:20] == second[:20] == f"dasp{hash_problem_id(PROMPT)}"
    assert hash_problem_id(PROMPT + [7, 8, 9, 50, 51]) != hash_problem_id(PROMPT)


@pytest.mark.asyncio
async def test_push_gate_suppresses_collection():
    client, _, store = make_client()
    token = DAS_PUSH_ENABLED.set(False)
    try:
        await client.generate("eval", prompt_ids=PROMPT, sampling_params={})
    finally:
        DAS_PUSH_ENABLED.reset(token)
    assert pushes(store) == 0
    await client.generate("train", prompt_ids=PROMPT, sampling_params={})
    assert pushes(store) == 1


@pytest.mark.asyncio
async def test_empty_response_is_not_pushed_and_store_absence_is_tolerated():
    server = FakeServer([TokenOutput(token_ids=[], stop_reason="completed", extra_fields={"global_steps": 1})])
    client, _, store = make_client(server=server)
    await client.generate("t", prompt_ids=PROMPT, sampling_params={})
    assert pushes(store) == 0
    # No store yet (ray.get_actor would fail): pushes are dropped silently.
    client._das_service = None
    client._das_push_client = None
    client._das_get_service = lambda: None
    await client.generate("t2", prompt_ids=PROMPT, sampling_params={})
    assert client._das_errors == 0


@pytest.mark.asyncio
async def test_full_determinism_keeps_verl_request_ids():
    client, server, _ = make_client(full_determinism=True)
    await client.generate("det-7", prompt_ids=PROMPT, sampling_params={})
    assert server.calls[0]["request_id"] == "det-7"


@pytest.mark.asyncio
async def test_wrap_is_idempotent_and_caches_subclass():
    client, _, _ = make_client()
    again = wrap_llm_client(client, DASConfig(enable=True))
    assert again is client
    other, _, _ = make_client()
    assert type(other) is type(client)
    assert type(client).__name__ == "DASLLMServerClient"


@pytest.mark.asyncio
async def test_fully_async_client_pushes_once_per_trajectory_and_signals_weight_versions():
    server = FakeServer(
        [
            TokenOutput(token_ids=[1, 2], stop_reason="aborted", extra_fields={"global_steps": 3}),
            TokenOutput(token_ids=[3, 4], stop_reason="completed", extra_fields={"global_steps": 4}),
        ]
    )
    client, server, store = make_client(FullyAsyncLLMServerClient, server=server, track_weight_version=True)
    assert isinstance(client, FullyAsyncLLMServerClient)
    out = await client.generate("traj", prompt_ids=PROMPT, sampling_params={"max_tokens": 16})
    assert out.token_ids == [1, 2, 3, 4]
    # Two engine calls (abort + resume), both carrying the hash, each with a fresh id.
    ids = [c["request_id"] for c in server.calls]
    assert len(ids) == 2 and ids[0] != ids[1]
    assert all(i.startswith(f"dasp{hash_problem_id(PROMPT)}-") for i in ids)
    assert server.calls[1]["prompt_ids"] == PROMPT + [1, 2]
    # One push with the concatenated response.
    assert pushes(store) == 1
    assert store._total_tokens == 4
    # First sighting of weight version 4 opens iteration 5.
    assert store.begin_iteration(0)["iteration"] == 5
    # Same or older versions do not re-signal.
    await client.generate("traj2", prompt_ids=PROMPT, sampling_params={"max_tokens": 16})
    assert store.begin_iteration(0)["iteration"] == 5
