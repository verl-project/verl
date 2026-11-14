# Copyright 2026 Individual Contributor: HaitaoWuTJU
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
"""Exercise the actual lifecycle without importing CUDA, vLLM, Ray or torch."""

import asyncio
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _load_module(name):
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(name, root / "verl/workers/rollout/vllm_rollout" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kv = _load_module("partial_kv")


@asynccontextmanager
async def _owner(tp=2):
    core = SimpleNamespace(
        outputs_queue=asyncio.Queue(),
        call_utility_async=AsyncMock(),
        set_weight_version_async=AsyncMock(),
    )
    engine = SimpleNamespace(
        engine_core=core,
        output_processor=SimpleNamespace(
            request_states={"internal": SimpleNamespace(external_req_id="request")},
            process_outputs=lambda *args, **kwargs: None,
        ),
        pause_generation=AsyncMock(),
        resume_generation=AsyncMock(),
        wait_for_requests_to_drain=AsyncMock(),
    )
    engine.output_handler = asyncio.create_task(core.outputs_queue.get())
    owner = SimpleNamespace(
        engine=engine,
        config=SimpleNamespace(tensor_model_parallel_size=tp, partial_rollout_kv_offload_headroom_gb=64),
        _admitting=0,
        _submission_paused=False,
        _rejecting=False,
        _resume_event=asyncio.Event(),
        global_steps=0,
    )
    owner._resume_event.set()
    lifecycle = kv.PartialKVOffload(owner)

    async def rpc(method, args=()):
        if method == "offload_partial_kv":
            return [dict(protocol=kv.PROTOCOL, kv_preserved=args[0], asleep=True) for _ in range(tp)]
        if method == "restore_partial_kv":
            return [
                dict(protocol=kv.PROTOCOL, kv_preserved=lifecycle.preserve, tags=args[0], runtime_restored=True)
                for _ in range(tp)
            ]
        return []

    engine.collective_rpc = AsyncMock(side_effect=rpc)
    lifecycle.install()
    await asyncio.sleep(0)
    try:
        yield owner, lifecycle
    finally:
        engine.output_handler.cancel()
        await asyncio.gather(engine.output_handler, return_exceptions=True)


def test_keep_train_refit_restore_and_resume():
    async def run():
        async with _owner() as (owner, lifecycle):
            await lifecycle.pause(True)
            assert owner._submission_paused and not owner._resume_event.is_set()
            owner.engine.pause_generation.assert_awaited_once_with(mode="keep", clear_cache=False)
            await lifecycle.sleep()
            owner.engine.wait_for_requests_to_drain.assert_not_awaited()
            assert lifecycle.state == "asleep"
            await lifecycle.wake(["weights"])
            assert lifecycle.state == "weights_awake"
            with pytest.raises(RuntimeError, match="refit"):
                await lifecycle.wake(["kv_cache"])
            await lifecycle.commit_version(1)
            owner.global_steps = 1
            owner.engine.engine_core.set_weight_version_async.assert_awaited_once_with("1")
            await lifecycle.wake(["kv_cache"])
            assert lifecycle.state == "awake" and lifecycle.paused
            owner.engine.resume_generation.assert_not_awaited()
            await lifecycle.resume()
            owner.engine.resume_generation.assert_awaited_once()
            # Opening admission is a separate all-replica operation.
            assert owner._submission_paused and not owner._resume_event.is_set()

    asyncio.run(run())


def test_initial_sleep_discards_and_native_repeat_calls_are_idempotent():
    async def run():
        async with _owner() as (owner, lifecycle):
            await lifecycle.drain()
            await lifecycle.sleep()
            await lifecycle.sleep()
            owner.engine.pause_generation.assert_awaited_once_with(mode="abort", clear_cache=True)
            copies = [
                call for call in owner.engine.collective_rpc.await_args_list if call.args[0] == "offload_partial_kv"
            ]
            assert len(copies) == 1
            await lifecycle.wake(["weights"])
            await lifecycle.wake(["weights"])
            await lifecycle.commit_version(0)
            await lifecycle.wake(["kv_cache"])
            await lifecycle.resume()
            await lifecycle.resume()
            owner.engine.resume_generation.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("state", ["asleep", "weights_awake", "failed"])
def test_incomplete_handoff_cannot_resume(state):
    async def run():
        async with _owner() as (owner, lifecycle):
            lifecycle.state = state
            lifecycle.paused = True
            with pytest.raises(RuntimeError):
                await lifecycle.resume()
            owner.engine.resume_generation.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["offload_partial_kv", "restore_partial_kv"])
@pytest.mark.parametrize("failure", ["missing_rank", "wrong_mode", "exception", "cancel"])
def test_uncertain_backup_or_restore_poison_the_handoff(operation, failure):
    async def run():
        async with _owner() as (owner, lifecycle):
            await lifecycle.pause(True)
            if operation == "restore_partial_kv":
                await lifecycle.sleep()
            original = owner.engine.collective_rpc.side_effect

            async def rpc(method, args=()):
                if method != operation:
                    return await original(method, args)
                if failure == "exception":
                    raise RuntimeError("copy failed")
                if failure == "cancel":
                    raise asyncio.CancelledError()
                rows = await original(method, args)
                if failure == "missing_rank":
                    return rows[:-1]
                rows[0]["kv_preserved"] = False
                return rows

            owner.engine.collective_rpc.side_effect = rpc
            with pytest.raises((RuntimeError, asyncio.CancelledError)):
                if operation == "offload_partial_kv":
                    await lifecycle.sleep()
                else:
                    await lifecycle.wake(["weights"])
            assert lifecycle.state == "failed"
            with pytest.raises(RuntimeError, match="restart"):
                await lifecycle.resume()
            owner.engine.resume_generation.assert_not_awaited()

    asyncio.run(run())


def test_pause_waits_for_admissions_and_fails_closed_on_timeout():
    async def run():
        async with _owner() as (owner, lifecycle):
            owner._admitting = 1
            pause = asyncio.create_task(lifecycle.pause(True))
            await asyncio.sleep(0)
            assert owner._submission_paused
            owner.engine.pause_generation.assert_not_awaited()
            owner._admitting = 0
            await pause
            owner.engine.pause_generation.assert_awaited_once()
        async with _owner() as (owner, lifecycle):
            owner._admitting = 1
            with pytest.raises(TimeoutError):
                await lifecycle.pause(True, timeout=0)
            assert lifecycle.state == "failed"
            owner.engine.pause_generation.assert_not_awaited()
            assert not owner._resume_event.is_set()

    asyncio.run(run())


def test_a_dequeued_output_batch_is_not_an_idle_fence():
    async def run():
        async with _owner() as (owner, lifecycle):
            owner.engine.output_handler.cancel()
            await asyncio.gather(owner.engine.output_handler, return_exceptions=True)
            owner.engine.output_handler = asyncio.create_task(asyncio.sleep(60))
            # Queue is empty, but no output-handler getter is waiting on it.
            with pytest.raises(TimeoutError, match="outputs"):
                await lifecycle.fence(timeout=0)

    asyncio.run(run())


def test_mode_cannot_change_against_unmapped_pools():
    async def run():
        async with _owner() as (_, lifecycle):
            await lifecycle.pause(True)
            await lifecycle.sleep()
            with pytest.raises(RuntimeError, match="preservation mode"):
                await lifecycle.pause(False)
            with pytest.raises(RuntimeError, match="refit"):
                await lifecycle.wake(["weights", "kv_cache"])

    asyncio.run(run())


def test_failed_version_publication_cannot_authorize_resume():
    async def run():
        async with _owner() as (owner, lifecycle):
            await lifecycle.pause(True)
            await lifecycle.sleep()
            await lifecycle.wake(["weights"])
            owner.engine.engine_core.set_weight_version_async.side_effect = RuntimeError("publication failed")
            with pytest.raises(RuntimeError, match="publication failed"):
                await lifecycle.commit_version(1)
            assert lifecycle.state == "failed" and not lifecycle.refitted
            with pytest.raises(RuntimeError, match="restart"):
                await lifecycle.resume()

    asyncio.run(run())


def test_native_output_version_wins_over_delayed_frontend_delivery():
    async def run():
        async with _owner() as (owner, lifecycle):
            trace = kv.NativeWeightVersions()
            lifecycle.traces["request"] = trace
            processor = owner.engine.output_processor
            owner.global_steps = 2
            processor.process_outputs(
                [SimpleNamespace(request_id="internal", new_token_ids=[8, 9])], weight_version="0"
            )
            processor.process_outputs([SimpleNamespace(request_id="internal", new_token_ids=[10])], weight_version="2")
            assert trace.finish([8, 9, 10]) == [
                dict(offset=0, length=2, policy_version=0),
                dict(offset=2, length=1, policy_version=2),
            ]
            assert trace.finish([8, 9]) == [dict(offset=0, length=2, policy_version=0)]
            with pytest.raises(RuntimeError, match="differs"):
                trace.finish([8, 11])

    asyncio.run(run())


@pytest.mark.parametrize("version", [None, 1, "unknown", "-1"])
def test_unstamped_native_tokens_are_rejected(version):
    with pytest.raises(RuntimeError, match="version"):
        kv.NativeWeightVersions().append([8], version)


def test_serialized_worker_replies_survive_repeated_observer_cancellation():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def utility(method, label):
            calls.append(label)
            if label == "observer":
                entered.set()
                await release.wait()
            return label

        core = SimpleNamespace(call_utility_async=utility)
        kv.serialize_worker_utilities(core)
        kv.serialize_worker_utilities(core)
        observer = asyncio.create_task(core.call_utility_async("collective_rpc", "observer"))
        await entered.wait()
        observer.cancel()
        await asyncio.sleep(0)
        observer.cancel()
        refit = asyncio.create_task(core.call_utility_async("collective_rpc", "refit"))
        await asyncio.sleep(0)
        assert calls == ["observer"]
        assert await core.call_utility_async("get_weight_version", "version") == "version"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert await refit == "refit"
        assert calls == ["observer", "version", "refit"]

    asyncio.run(run())


@pytest.mark.parametrize("step,expected", [(1, True), (2, True), (3, False), (4, False), (5, True), (6, False)])
def test_checkpoint_validation_and_final_boundaries_discard(step, expected):
    assert kv.keep_partial_requests(step, total_steps=6, save_freq=3, test_freq=4) is expected
