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

Long-term-care minimum: a recognized engine failure triggers CAS removal
plus a bounded re-route to a recovered server, the retry budget is
enforced, and a disabled restart switch keeps the pre-PR behavior exactly.
Real Ray actors; ``EngineDeadError`` is faked by name — the client
classifies engine failures by class name so it stays vllm-import-free.
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


def _client_config(enabled=True):
    restart = SimpleNamespace(enabled=enabled)
    return SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(restart=restart, name="vllm")))


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
    acquired (or fenced out) — deterministic either way."""
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
            await asyncio.wait_for(client.generate("req-3", prompt_ids=[1], sampling_params={}), timeout=30)
        except RuntimeError as exc:
            raised = exc
        await waiter
        assert raised is not None and "engine recovery gave up" in str(raised)
        assert ray.get(lb.get_all_servers.remote()) == []

    asyncio.run(scenario())


def test_restart_disabled_keeps_current_behavior():
    ray.init(ignore_reinit_error=True)

    async def scenario():
        server = _FlakyServer.remote([_receipt_output()])
        lb = _make_lb({"solo": server})
        client = LLMServerClient(config=_client_config(enabled=False), load_balancer_handle=lb)

        output = await client.generate("req-5", prompt_ids=[1], sampling_params={})
        assert output.stop_reason == "engine_failed"
        # Restart disabled: no receipt flag, no interception, no fencing.
        assert ray.get(server.get_recover_flags.remote()) == ["absent"]
        assert ray.get(lb.get_all_servers.remote()) == ["solo"]

    asyncio.run(scenario())
