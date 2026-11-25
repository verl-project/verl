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
"""CPU tests for the client bounded replay (verl/workers/rollout/llm_server.py).

Real Ray actors for the load balancer and the servers. ``EngineDeadError`` is
faked by name — the client classifies engine failures by class name precisely
so it stays vllm-import-free. Determinism pattern: the LB starts with the
failing server only; a side task registers the healthy server after it has
actually been acquired, so the bounded pool-wait is guaranteed to observe the
recovery transition. The client reports nothing — fencing + replay is
the whole request-path self-healing (detection is the manager's poll).
"""

import asyncio
from types import SimpleNamespace

import ray

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.router import GlobalRequestLoadBalancer


class EngineDeadError(Exception):
    """Name-matched stand-in for vllm's EngineDeadError (client classifies by name)."""


def _good_output(tokens=(1, 2, 3)):
    return TokenOutput(token_ids=list(tokens), log_probs=None, stop_reason="completed")


def _receipt_output():
    return TokenOutput(
        token_ids=[7],
        log_probs=None,
        stop_reason="engine_failed",
        extra_fields={"engine_failed": True, "global_steps": 5},
    )


def _client_config(enabled=True, name="vllm"):
    restart = SimpleNamespace(enabled=enabled)
    return SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(restart=restart, name=name)))


@ray.remote
class _FlakyServer:
    def __init__(self, script):
        self.script = list(script)
        self.recover_flags = []

    async def generate(self, *, request_id, prompt_ids, sampling_params, session_id=None, **kwargs):
        self.recover_flags.append(kwargs.get("recover_engine_failure", "absent"))
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    def get_recover_flags(self):
        return list(self.recover_flags)


def _make_lb(servers):
    return ray.remote(GlobalRequestLoadBalancer).remote(servers=servers)


async def _add_server_after_acquire(lb, address, handle, wait_address: str):
    """Register the healthy server once the failing one has actually been
    acquired — or once it has been fenced out, whichever comes first (the
    acquire->remove window can be milliseconds wide). Deterministic either
    way: the healthy server never races the client's first acquire, and the
    failing server is the pool's only member until then."""
    while True:
        status = await lb.get_status.remote()
        inflight = status["servers"].get(wait_address)
        if inflight is None or inflight >= 1:
            break
        await asyncio.sleep(0.05)
    await lb.add_servers.remote({address: handle})


def test_receipt_triggers_cas_removal_and_replay():
    ray.init(ignore_reinit_error=True)

    async def scenario():
        flaky = _FlakyServer.remote([_receipt_output()])
        good = _FlakyServer.remote([_good_output()])
        lb = _make_lb({"flaky": flaky})
        client = LLMServerClient(config=_client_config(), load_balancer_handle=lb)
        waiter = asyncio.create_task(_add_server_after_acquire(lb, "good", good, "flaky"))

        output = await client.generate("req-1", prompt_ids=[1], sampling_params={})
        await waiter

        assert output.token_ids == [1, 2, 3]
        assert output.stop_reason == "completed"
        assert ray.get(flaky.get_recover_flags.remote()) == [True]
        assert ray.get(good.get_recover_flags.remote()) == [True]
        assert ray.get(lb.get_all_servers.remote()) == ["good"]

    asyncio.run(scenario())


def test_engine_exception_triggers_replay_to_recovered_server():
    ray.init(ignore_reinit_error=True)

    async def scenario():
        flaky = _FlakyServer.remote([EngineDeadError("core died")])
        good = _FlakyServer.remote([_good_output()])
        lb = _make_lb({"flaky": flaky})
        client = LLMServerClient(config=_client_config(), load_balancer_handle=lb)
        waiter = asyncio.create_task(_add_server_after_acquire(lb, "good", good, "flaky"))

        output = await client.generate("req-2", prompt_ids=[1], sampling_params={})
        await waiter

        assert output.token_ids == [1, 2, 3]
        assert ray.get(lb.get_all_servers.remote()) == ["good"]

    asyncio.run(scenario())


def test_killed_actor_is_detected_as_engine_failure():
    ray.init(ignore_reinit_error=True)

    async def scenario():
        victim = _FlakyServer.remote([_good_output()])
        survivor = _FlakyServer.remote([_good_output()])
        lb = _make_lb({"victim": victim})
        client = LLMServerClient(config=_client_config(), load_balancer_handle=lb)
        waiter = asyncio.create_task(_add_server_after_acquire(lb, "survivor", survivor, "victim"))

        ray.kill(victim, no_restart=True)
        output = await client.generate("req-3", prompt_ids=[1], sampling_params={})
        await waiter

        assert output.token_ids == [1, 2, 3]
        assert ray.get(lb.get_all_servers.remote()) == ["survivor"]

    asyncio.run(scenario())


def test_retry_budget_exhaustion_stops_the_replay(monkeypatch):
    ray.init(ignore_reinit_error=True)
    monkeypatch.setattr("verl.workers.rollout.fault_tolerance.MAX_ENGINE_RETRIES", 1)

    async def scenario():
        first = _FlakyServer.remote([_receipt_output()])
        second = _FlakyServer.remote([_receipt_output()])
        lb = _make_lb({"first": first})
        client = LLMServerClient(config=_client_config(), load_balancer_handle=lb)
        waiter = asyncio.create_task(_add_server_after_acquire(lb, "second", second, "first"))

        raised = None
        try:
            await asyncio.wait_for(client.generate("req-4", prompt_ids=[1], sampling_params={}), timeout=30)
        except RuntimeError as exc:
            raised = exc
        await waiter
        assert raised is not None and "engine recovery gave up" in str(raised)
        assert ray.get(lb.get_all_servers.remote()) == []

    asyncio.run(scenario())


def test_non_engine_error_propagates_without_replay():
    ray.init(ignore_reinit_error=True)

    async def scenario():
        bad_request = _FlakyServer.remote([ValueError("bad sampling params")])
        lb = _make_lb({"bad": bad_request})
        client = LLMServerClient(config=_client_config(), load_balancer_handle=lb)
        raised = None
        try:
            await asyncio.wait_for(client.generate("req-5", prompt_ids=[1], sampling_params={}), timeout=30)
        except ValueError as exc:
            raised = exc
        assert raised is not None and "bad sampling params" in str(raised)
        assert ray.get(lb.get_all_servers.remote()) == ["bad"]
        assert ray.get(bad_request.get_recover_flags.remote()) == [True]

    asyncio.run(scenario())


def test_restart_disabled_keeps_current_behavior():
    ray.init(ignore_reinit_error=True)

    async def scenario():
        server = _FlakyServer.remote([_receipt_output()])
        lb = _make_lb({"solo": server})
        client = LLMServerClient(config=_client_config(enabled=False), load_balancer_handle=lb)

        output = await client.generate("req-6", prompt_ids=[1], sampling_params={})
        assert output.stop_reason == "engine_failed"
        # Restart disabled: the receipt flag is never sent to the server and the
        # receipt-shaped output is returned as-is (no interception, no fencing).
        assert ray.get(server.get_recover_flags.remote()) == ["absent"]
        assert ray.get(lb.get_all_servers.remote()) == ["solo"]

        dead = _FlakyServer.remote([EngineDeadError("core died")])
        lb2 = _make_lb({"solo2": dead})
        client2 = LLMServerClient(config=_client_config(enabled=False), load_balancer_handle=lb2)
        raised = None
        try:
            await asyncio.wait_for(client2.generate("req-7", prompt_ids=[1], sampling_params={}), timeout=30)
        except Exception as exc:
            raised = exc
        assert raised is not None
        assert ray.get(lb2.get_all_servers.remote()) == ["solo2"]

    asyncio.run(scenario())


def test_receipt_incapable_server_never_receives_the_receipt_flag():
    ray.init(ignore_reinit_error=True)

    async def scenario():
        server = _FlakyServer.remote([_good_output()])
        lb = _make_lb({"sg": server})
        client = LLMServerClient(config=_client_config(name="trtllm"), load_balancer_handle=lb)
        await client.generate("req-8", prompt_ids=[1], sampling_params={})
        assert ray.get(server.get_recover_flags.remote()) == ["absent"]

    asyncio.run(scenario())


def test_sglang_server_does_not_receive_the_receipt_flag():
    ray.init(ignore_reinit_error=True)

    async def scenario():
        server = _FlakyServer.remote([_good_output()])
        lb = _make_lb({"sg": server})
        client = LLMServerClient(config=_client_config(name="sglang"), load_balancer_handle=lb)
        await client.generate("req-9", prompt_ids=[1], sampling_params={})
        # The SGLang server-side restart contract (receipt-aware generate) is not
        # implemented yet, so sglang stays receipt-incapable until it lands.
        assert ray.get(server.get_recover_flags.remote()) == ["absent"]

    asyncio.run(scenario())
