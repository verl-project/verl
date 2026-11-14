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
"""Late requests must leave a deactivated replica instead of waiting for its GPU."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from verl.experimental.fully_async_policy.dynamic_schedule import dynamic_resource_controller
from verl.workers.config import RolloutConfig
from verl.workers.rollout.sglang_rollout import async_sglang_server


class _TokenizerManager:
    def __init__(self):
        self.generate_calls = 0
        self.pause_error = self.resume_error = None
        self.pause_wait = None
        self.pause_entered = asyncio.Event()
        self.resume_calls = 0

    async def generate_request(self, obj, request):
        self.generate_calls += 1
        yield {"output_ids": [7, 8], "meta_info": {"finish_reason": {"type": "length"}}}

    async def pause_generation(self, obj):
        self.pause_entered.set()
        if self.pause_wait is not None:
            await self.pause_wait.wait()
        if self.pause_error is not None:
            raise self.pause_error

    async def continue_generation(self, obj):
        self.resume_calls += 1
        if self.resume_error is not None:
            raise self.resume_error


def _server(monkeypatch):
    monkeypatch.setattr(
        async_sglang_server.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_actor_name=lambda: "submission-test"),
    )
    server = object.__new__(async_sglang_server.SGLangHttpServer)
    server.node_rank, server.global_steps = 0, 7
    server.config = RolloutConfig(name="sglang", max_model_len=1024, prompt_length=512, response_length=16)
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server._disaggregation_role, server._pd_decode_peers = "null", []
    server.tokenizer_manager = _TokenizerManager()
    server._submission_paused = server._rejecting = False
    server._resume_event = asyncio.Event()
    server._resume_event.set()
    server._generation_control_lock = asyncio.Lock()
    return server


async def _generate(server):
    return await server.generate([1, 2, 3], {"max_new_tokens": 2}, "late-request")


async def _settle():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def test_deactivated_server_rejects_before_entering_tokenizer_manager(monkeypatch):
    async def run():
        server = _server(monkeypatch)
        await server.abort_all_requests(reject_request=True)
        output = await asyncio.wait_for(_generate(server), 1)
        assert output.stop_reason == "aborted"
        assert output.token_ids == []
        assert output.extra_fields["global_steps"] == 7
        assert server.tokenizer_manager.generate_calls == 0

    asyncio.run(run())


def test_temporary_pause_parks_outside_native_manager_until_resume(monkeypatch):
    async def run():
        server = _server(monkeypatch)
        await server.abort_all_requests()
        task = asyncio.create_task(_generate(server))
        await _settle()
        assert not task.done()
        assert server.tokenizer_manager.generate_calls == 0
        await server.resume_generation()
        assert (await asyncio.wait_for(task, 1)).token_ids == [7, 8]

    asyncio.run(run())


def test_deactivation_wakes_requests_parked_by_an_earlier_temporary_pause(monkeypatch):
    async def run():
        server = _server(monkeypatch)
        await server.abort_all_requests()
        task = asyncio.create_task(_generate(server))
        await _settle()
        await server.abort_all_requests(reject_request=True)
        assert (await asyncio.wait_for(task, 1)).stop_reason == "aborted"
        assert server.tokenizer_manager.generate_calls == 0

    asyncio.run(run())


def test_new_temporary_pause_restores_parking_after_a_deactivation(monkeypatch):
    async def run():
        server = _server(monkeypatch)
        await server.abort_all_requests(reject_request=True)
        await server.abort_all_requests()
        task = asyncio.create_task(_generate(server))
        await _settle()
        assert not task.done()
        await server.resume_generation()
        assert (await asyncio.wait_for(task, 1)).token_ids == [7, 8]

    asyncio.run(run())


def test_failed_pause_keeps_requests_out_until_explicit_resume(monkeypatch):
    async def run():
        server = _server(monkeypatch)
        server.tokenizer_manager.pause_error = RuntimeError("pause failed")
        with pytest.raises(RuntimeError, match="pause failed"):
            await server.abort_all_requests()
        assert (await _generate(server)).stop_reason == "aborted"
        assert server.tokenizer_manager.generate_calls == 0
        await server.resume_generation()
        assert (await _generate(server)).token_ids == [7, 8]

    asyncio.run(run())


def test_failed_resume_does_not_reopen_admission(monkeypatch):
    async def run():
        server = _server(monkeypatch)
        await server.abort_all_requests(reject_request=True)
        server.tokenizer_manager.resume_error = RuntimeError("resume failed")
        with pytest.raises(RuntimeError, match="resume failed"):
            await server.resume_generation()
        assert (await _generate(server)).stop_reason == "aborted"
        assert server.tokenizer_manager.generate_calls == 0

    asyncio.run(run())


def test_resume_waits_for_an_in_progress_pause_control_call(monkeypatch):
    async def run():
        server = _server(monkeypatch)
        server.tokenizer_manager.pause_wait = asyncio.Event()
        pause = asyncio.create_task(server.abort_all_requests())
        await server.tokenizer_manager.pause_entered.wait()
        resume = asyncio.create_task(server.resume_generation())
        await _settle()
        assert server.tokenizer_manager.resume_calls == 0
        server.tokenizer_manager.pause_wait.set()
        await asyncio.wait_for(asyncio.gather(pause, resume), 1)
        assert (await _generate(server)).token_ids == [7, 8]

    asyncio.run(run())


@pytest.mark.parametrize("reject_request", [False, True])
def test_replica_forwards_rejection_only_to_its_primary_server(reject_request):
    async def run():
        calls = []

        async def abort(**kwargs):
            calls.append(kwargs)

        replica = object.__new__(async_sglang_server.SGLangReplica)
        replica.servers = [SimpleNamespace(abort_all_requests=SimpleNamespace(remote=abort)), object()]
        await replica.abort_all_requests(reject_request=reject_request)
        assert calls == [{"reject_request": reject_request}]

    asyncio.run(run())


def test_dynamic_deactivation_removes_routing_then_rejects_then_sleeps(monkeypatch):
    async def run():
        calls = []

        async def remove(ids):
            calls.append(("remove", ids))

        async def abort(reject_request=False):
            calls.append(("abort", reject_request))

        async def sleep():
            calls.append(("sleep",))

        monkeypatch.setattr(dynamic_resource_controller.ray, "get", lambda value: value)
        controller = object.__new__(dynamic_resource_controller.DynamicResourceController)
        controller.rollouter = SimpleNamespace(
            get_all_hybrid_replicas=SimpleNamespace(remote=lambda: {"a": object()}),
            remove_replicas=SimpleNamespace(remote=remove),
        )
        controller.hybrid_checkpoint_manager = SimpleNamespace(abort_replicas=abort, sleep_replicas=sleep)
        controller.deactivate_count = 0
        await controller.deactivate_hybrid_replicas(global_steps=7)
        assert calls == [("remove", ["a"]), ("abort", True), ("sleep",)]

    asyncio.run(run())


def _v1_trainer(monkeypatch, server, calls):
    from verl.checkpoint_engine import CheckpointEngineManager
    from verl.trainer.ppo.v1 import trainer_separate_async

    def remove(server_ids):
        calls.append(("remove", server_ids))

    async def abort(**kwargs):
        calls.append(("abort", kwargs["reject_request"]))
        await server.abort_all_requests(**kwargs)

    async def sleep():
        output = await asyncio.wait_for(_generate(server), 1)
        calls.append(("sleep", output.stop_reason))

    replica = object.__new__(async_sglang_server.SGLangReplica)
    replica.servers = [SimpleNamespace(abort_all_requests=SimpleNamespace(remote=abort))]
    replica.sleep = sleep
    manager = object.__new__(CheckpointEngineManager)
    manager.replicas = [replica]
    trainer = object.__new__(trainer_separate_async.PPOTrainerSeparateAsync)
    trainer._enable_hybrid_replicas = True
    trainer.current_mode = trainer_separate_async.HybridEngineMode.ROLLOUT
    trainer.checkpoint_manager = manager
    trainer.llm_server_manager = SimpleNamespace(server_addresses=["hybrid"])
    trainer.standalone_server_manager = SimpleNamespace(
        global_load_balancer=SimpleNamespace(remove_servers=SimpleNamespace(remote=remove))
    )
    monkeypatch.setattr(trainer_separate_async.ray, "get", lambda value: value)
    return trainer


def test_v1_switch_rejects_a_late_request_before_releasing_hybrid_memory(monkeypatch):
    server, calls = _server(monkeypatch), []
    trainer = _v1_trainer(monkeypatch, server, calls)

    trainer.switch_to_trainer()

    assert calls == [("remove", ["hybrid"]), ("abort", True), ("sleep", "aborted")]
    assert trainer.current_mode.value == "trainer"
    assert server.tokenizer_manager.generate_calls == 0


def test_v1_failed_pause_does_not_release_memory_or_readmit_requests(monkeypatch):
    server, calls = _server(monkeypatch), []
    server.tokenizer_manager.pause_error = RuntimeError("native pause failed")
    trainer = _v1_trainer(monkeypatch, server, calls)
    original_mode = trainer.current_mode

    with pytest.raises(RuntimeError, match="native pause failed"):
        trainer.switch_to_trainer()

    assert calls == [("remove", ["hybrid"]), ("abort", True)]
    assert trainer.current_mode == original_mode
    assert asyncio.run(_generate(server)).stop_reason == "aborted"
    assert server.tokenizer_manager.generate_calls == 0


def test_v1_switch_without_hybrid_replicas_does_not_pause_or_sleep(monkeypatch):
    server, calls = _server(monkeypatch), []
    trainer = _v1_trainer(monkeypatch, server, calls)
    trainer._enable_hybrid_replicas = False

    trainer.switch_to_trainer()

    assert calls == []
    assert server.tokenizer_manager.generate_calls == 0
