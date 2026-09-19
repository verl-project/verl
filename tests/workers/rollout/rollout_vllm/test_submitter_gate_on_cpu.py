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
"""GPU-free tests for the vLLMHttpServer submission gate.

vLLM's pause stops requests being scheduled but still accepts them. A request
admitted between abort_all_requests() and resume_generation() is parked in the
scheduler's waiting queue and masked out of the drain's liveness check, so
wait_for_requests_to_drain() cannot return. These tests pin the ordering that
makes such an admission impossible.

They also pin abort_all_requests(reject_request=True), which fails late arrivals
instead of parking them when the server is leaving the load balancer and no
resume_generation() is coming soon.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeEngine:
    """Records the state of the gate at the moment the engine is paused."""

    def __init__(self):
        self.output_processor = SimpleNamespace(request_states={})
        self.server = None
        self.pause_calls = 0
        self.resume_calls = 0
        self.admitting_at_pause = None

    async def pause_generation(self, **kwargs):
        self.pause_calls += 1
        self.admitting_at_pause = self.server._admitting

    async def resume_generation(self):
        self.resume_calls += 1


def _make_server(node_rank: int = 0):
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.node_rank = node_rank
    server.global_steps = 7
    server.engine = _FakeEngine()
    server.engine.server = server
    server._submission_paused = False
    server._admitting = 0
    server._resume_event = asyncio.Event()
    server._resume_event.set()
    server._rejecting = False
    return server


def test_abort_does_not_pause_until_inflight_admissions_land():
    async def main():
        server = _make_server()
        server._admitting = 1  # a turn is past the gate but not yet in the engine

        abort = asyncio.create_task(server.abort_all_requests())
        await asyncio.sleep(0.05)

        assert server._submission_paused is True, "gate must close before the barrier runs"
        assert not abort.done(), "abort must not proceed while an admission is in flight"
        assert server.engine.pause_calls == 0, "engine paused while an admission was in flight"

        server._admitting = 0  # the in-flight admission reaches the engine
        await asyncio.wait_for(abort, timeout=5)

        assert server.engine.pause_calls == 1
        assert server.engine.admitting_at_pause == 0

    asyncio.run(main())


def test_submission_parks_while_gate_closed_and_wakes_on_resume():
    async def main():
        server = _make_server()
        await server.abort_all_requests()
        assert server._submission_paused is True

        task = asyncio.create_task(server._park_until_admitted("r1"))
        await asyncio.sleep(0.05)
        assert not task.done(), "submission must park while the gate is closed"
        assert server._admitting == 0

        await server.resume_generation()
        assert await asyncio.wait_for(task, timeout=5) is None
        assert server._admitting == 1

    asyncio.run(main())


def test_reject_request_fails_late_arrivals_instead_of_parking():
    async def main():
        server = _make_server()
        await server.abort_all_requests(reject_request=True)

        output = await asyncio.wait_for(server._park_until_admitted("late"), timeout=5)

        assert output.stop_reason == "aborted", "a rejecting gate must fail over, not park"
        assert output.token_ids == []
        assert output.extra_fields["global_steps"] == 7
        assert server._admitting == 0, "rejected requests never count as admissions"
        assert server._submission_paused is True, "the gate stays closed until resume_generation"

    asyncio.run(main())


def test_weight_sync_abort_restores_parking_after_a_rejecting_abort():
    # switch_to_trainer aborts with reject_request=True; the weight sync inside the following
    # switch_to_rollout aborts again with the default, and by then a resume is imminent, so
    # requests must go back to parking rather than being failed over.
    async def main():
        server = _make_server()
        await server.abort_all_requests(reject_request=True)
        assert server._rejecting is True

        await server.abort_all_requests()
        assert server._rejecting is False

        task = asyncio.create_task(server._park_until_admitted("r1"))
        await asyncio.sleep(0.05)
        assert not task.done(), "a plain abort must restore parking"

        await server.resume_generation()
        assert await asyncio.wait_for(task, timeout=5) is None

    asyncio.run(main())


def test_resume_clears_rejection():
    async def main():
        server = _make_server()
        await server.abort_all_requests(reject_request=True)

        await server.resume_generation()

        assert server._rejecting is False
        assert server._submission_paused is False
        assert await server._park_until_admitted("r1") is None
        assert server._admitting == 1

    asyncio.run(main())


def test_resume_reopens_gate_on_non_head_server():
    async def main():
        server = _make_server(node_rank=1)
        server._submission_paused = True
        server._resume_event.clear()

        await server.resume_generation()

        assert server._submission_paused is False, "non-head server stays gated forever"
        assert server._resume_event.is_set()
        assert server.engine.resume_calls == 0, "only node rank 0 drives the engine"

    asyncio.run(main())


def test_resume_on_head_server_also_resumes_engine():
    async def main():
        server = _make_server(node_rank=0)
        await server.abort_all_requests()

        await server.resume_generation()

        assert server._submission_paused is False
        assert server._resume_event.is_set()
        assert server.engine.resume_calls == 1

    asyncio.run(main())


def test_barrier_times_out_instead_of_hanging(monkeypatch):
    # raising=False: this asserts the barrier cannot deadlock, not that the constant exists.
    monkeypatch.setattr(vllm_async_server, "_GATE_BARRIER_TIMEOUT_S", 0.05, raising=False)

    async def main():
        server = _make_server()
        server._admitting = 1  # never clears

        await asyncio.wait_for(server.abort_all_requests(), timeout=5)

        assert server.engine.pause_calls == 1, "barrier must proceed rather than deadlock"

    asyncio.run(main())
