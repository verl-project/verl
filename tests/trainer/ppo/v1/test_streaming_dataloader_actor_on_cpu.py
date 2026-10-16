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
import inspect
from collections import defaultdict
from contextlib import suppress
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.streaming_dataloader as streaming_module
import verl.trainer.ppo.v1.trainer_base as trainer_module
from verl.trainer.ppo.v1.streaming_dataloader import StreamingDataloader, StreamingDataloaderAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def _sync_stub():
    return object.__new__(StreamingDataloader)


def _stub_methods(actor, **methods):
    for name, value in methods.items():
        setattr(actor, name, value)


def _async_stub():
    actor = object.__new__(StreamingDataloaderAsync)
    actor._validation_active = False
    return actor


async def _advance_and_maintain_watermark(actor):
    if actor._advance_once():
        actor._refill_to_watermark()


def _replay_state(*, pending=(), running=(), finished=(), failure=()):
    replay = SimpleNamespace(
        pending_keys=defaultdict(set),
        running_keys=defaultdict(set),
        finished_keys=defaultdict(set),
        failure_keys=defaultdict(set),
    )
    replay.pending_keys["train"].update(pending)
    replay.running_keys["train"].update(running)
    replay.finished_keys["train"].update(finished)
    replay.failure_keys["train"].update(failure)
    return replay


def test_async_dataloader_inherits_sync_dataloader():
    assert issubclass(StreamingDataloaderAsync, StreamingDataloader)


def _sampler_config(trainer_mode, custom_sampler=None):
    return OmegaConf.create(
        {"trainer": {"v1": {"trainer_mode": trainer_mode, "sampler": {"custom_sampler": custom_sampler}}}}
    )


@pytest.mark.parametrize(
    ("trainer_mode", "expected_cls"),
    [("sync", StreamingDataloader), ("colocate_async", StreamingDataloaderAsync)],
)
def test_default_dataloader_class_matches_trainer_mode(trainer_mode, expected_cls):
    assert streaming_module.resolve_streaming_dataloader_cls(_sampler_config(trainer_mode)) is expected_cls


def test_custom_sampler_subclasses_mode_dataloader(monkeypatch):
    class CustomAsyncDataloader(StreamingDataloaderAsync):
        pass

    monkeypatch.setattr(streaming_module, "load_extern_type", lambda *_args: CustomAsyncDataloader)
    config = _sampler_config("separate_async", {"path": "custom.py", "name": "CustomAsyncDataloader"})

    assert streaming_module.resolve_streaming_dataloader_cls(config) is CustomAsyncDataloader


@pytest.mark.parametrize(
    ("trainer_mode", "custom_cls"),
    [("sync", StreamingDataloaderAsync), ("colocate_async", StreamingDataloader), ("sync", object)],
)
def test_custom_sampler_rejects_classes_for_other_modes(monkeypatch, trainer_mode, custom_cls):
    monkeypatch.setattr(streaming_module, "load_extern_type", lambda *_args: custom_cls)
    config = _sampler_config(trainer_mode, {"path": "custom.py", "name": "Custom"})

    with pytest.raises(TypeError, match="must subclass"):
        streaming_module.resolve_streaming_dataloader_cls(config)


def test_session_watermark_refills_complete_prompt_groups():
    actor = _async_stub()
    actor.watermark_sessions = 5
    actor.default_rollout_n = 2
    actor.gen_batch_size = 1
    actor.max_outstanding_groups = None
    _stub_methods(actor, unfinished_session_count=lambda *_args: 1)
    requested = []
    actor._refill_exact = lambda count: requested.append(count) or count

    assert actor._refill_to_watermark() == 2

    assert requested == [2]


@pytest.mark.parametrize(("max_outstanding_groups", "expected"), [(5, 1), (4, 0), (None, 3)])
def test_outstanding_cap_limits_watermark_refill(max_outstanding_groups, expected):
    actor = _async_stub()
    actor.watermark_sessions = 8
    actor.default_rollout_n = 2
    actor.max_outstanding_groups = max_outstanding_groups
    replay = _replay_state(running={"r0"}, finished={"f0", "f1", "f2"})
    _stub_methods(
        actor,
        pending_keys=replay.pending_keys,
        running_keys=replay.running_keys,
        finished_keys=replay.finished_keys,
        failure_keys=replay.failure_keys,
        unfinished_session_count=lambda *_args: 2,
    )
    requested = []
    actor._refill_exact = lambda count: requested.append(count) or count

    assert actor.outstanding_group_count("train") == 4
    assert actor._refill_to_watermark() == expected
    assert requested == ([expected] if expected else [])


def test_session_watermark_rejects_positive_value_that_rounds_to_zero():
    actor = _async_stub()
    actor.trainer_mode = "colocate_async"
    actor.max_off_policy_strategy = "drop"
    actor.max_off_policy_threshold = 8
    actor.config = OmegaConf.create(
        {
            "trainer": {"v1": {"colocate_async": {"watermark_batches": 0.5}}},
            "data": {"train_batch_size": 1},
            "actor_rollout_ref": {"rollout": {"n": 1}},
        }
    )

    with pytest.raises(ValueError, match="resolves to zero"):
        actor._init_refill_policy()


@pytest.mark.parametrize(("threshold", "rejected"), [(1, True), (2, False)])
def test_async_wait_strategy_requires_threshold_above_one(threshold, rejected):
    actor = _async_stub()
    actor.trainer_mode = "colocate_async"
    actor.max_off_policy_strategy = "wait"
    actor.max_off_policy_threshold = threshold
    actor.config = OmegaConf.create(
        {
            "trainer": {"v1": {"colocate_async": {"watermark_batches": 1}}},
            "data": {"train_batch_size": 2},
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )

    if rejected:
        with pytest.raises(ValueError, match="requires max_off_policy_threshold >= 2"):
            actor._init_refill_policy()
    else:
        actor._init_refill_policy()
        assert actor.watermark_sessions == 4


def _refill_policy_stub(*, watermark_batches=2, max_outstanding_batches=4, threshold=8, trainer_mode="separate_async"):
    actor = _async_stub()
    actor.trainer_mode = trainer_mode
    actor.max_off_policy_strategy = "drop"
    actor.max_off_policy_threshold = threshold
    actor.config = OmegaConf.create(
        {
            "trainer": {
                "v1": {
                    trainer_mode: {
                        "watermark_batches": watermark_batches,
                        "max_outstanding_batches": max_outstanding_batches,
                    }
                }
            },
            "data": {"train_batch_size": 8},
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    return actor


@pytest.mark.parametrize(("max_outstanding_batches", "expected"), [(4, 32), (2.5, 20), (None, None)])
def test_outstanding_cap_resolves_to_prompt_groups(max_outstanding_batches, expected):
    actor = _refill_policy_stub(max_outstanding_batches=max_outstanding_batches)
    actor._init_refill_policy()
    assert actor.max_outstanding_groups == expected


def test_colocate_async_has_no_outstanding_cap():
    actor = _refill_policy_stub(trainer_mode="colocate_async", watermark_batches=3, max_outstanding_batches=2)
    actor._init_refill_policy()
    assert actor.max_outstanding_groups is None


def test_outstanding_cap_must_cover_watermark():
    actor = _refill_policy_stub(watermark_batches=3, max_outstanding_batches=2)
    with pytest.raises(ValueError, match="must be >= watermark_batches"):
        actor._init_refill_policy()


def test_outstanding_cap_warns_when_staleness_can_reach_threshold(monkeypatch):
    warnings = []
    monkeypatch.setattr(streaming_module.logger, "warning", warnings.append)

    _refill_policy_stub(max_outstanding_batches=4, threshold=5)._init_refill_policy()
    assert warnings == []
    _refill_policy_stub(max_outstanding_batches=4, threshold=4)._init_refill_policy()
    assert len(warnings) == 1 and "max_off_policy_threshold=4" in warnings[0]


def _gen_batch_config(trainer_mode, gen_batch_size, *, dapo=False, refill_failed=False):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": 4, "gen_batch_size": gen_batch_size},
            "algorithm": {"filter_groups": {"enable": dapo}},
            "trainer": {"v1": {"trainer_mode": trainer_mode, "sampler": {"sync_refill_failed_groups": refill_failed}}},
        }
    )


@pytest.mark.parametrize(
    ("trainer_mode", "configured", "dapo", "refill_failed", "expected"),
    [
        ("sync", None, False, False, 4),
        ("sync", 1, False, False, 1),
        ("sync", 4, False, False, 4),
        ("sync", None, True, False, 1),
        ("sync", None, False, True, 1),
        ("colocate_async", None, False, False, 1),
        ("separate_async", 1, False, False, 1),
    ],
)
def test_resolve_gen_batch_size(trainer_mode, configured, dapo, refill_failed, expected):
    config = _gen_batch_config(trainer_mode, configured, dapo=dapo, refill_failed=refill_failed)
    assert streaming_module.resolve_gen_batch_size(config) == expected


@pytest.mark.parametrize(
    ("trainer_mode", "dapo", "refill_failed"),
    [("separate_async", False, False), ("sync", True, False), ("sync", False, True)],
)
def test_resolve_gen_batch_size_overrides_exact_refill_modes_to_one(monkeypatch, trainer_mode, dapo, refill_failed):
    warnings = []
    monkeypatch.setattr(streaming_module.logger, "warning", warnings.append)
    config = _gen_batch_config(trainer_mode, 2, dapo=dapo, refill_failed=refill_failed)

    assert streaming_module.resolve_gen_batch_size(config) == 1
    assert warnings == ["data.gen_batch_size=2 is overridden to 1."]


def test_resolve_gen_batch_size_rejects_sync_value_between_one_and_train_batch():
    with pytest.raises(ValueError, match="gen_batch_size to be 1 or train_batch_size"):
        streaming_module.resolve_gen_batch_size(_gen_batch_config("sync", 2))


def _sync_coordinator_stub():
    actor = _sync_stub()
    actor.global_steps = 0
    actor._init_replay_state(
        max_off_policy_threshold=8, max_off_policy_strategy="drop", sampler_kwargs={}, poll_interval=0.01
    )
    return actor


def test_sync_sample_submits_batch_then_waits_for_coordinator():
    async def run():
        actor = _sync_coordinator_stub()
        events = []
        actor._refill_exact = lambda count: events.append(("refill", count)) or count

        def advance(request):
            events.append(("poll", request.global_steps, request.batch_size))
            return ["a", "b"], {"sample": 1}, 2

        actor._advance_request = advance

        result = await asyncio.wait_for(actor.sample(global_steps=3, batch_size=2), timeout=1)

        assert result == (["a", "b"], {"sample": 1}, 2)
        assert events == [("refill", 2), ("poll", 3, 2)]
        assert actor._pending_sample is None
        actor._coordinator_task.cancel()

    asyncio.run(run())


def test_inventory_queries_exist_only_on_async_dataloader():
    for method in ("count_sampleable", "wait_sampleable"):
        assert not hasattr(StreamingDataloader, method)
        assert inspect.iscoroutinefunction(getattr(StreamingDataloaderAsync, method))


def test_sync_coordinator_idles_without_pending_request():
    actor = _sync_coordinator_stub()
    actor._advance_request = lambda _request: pytest.fail("sync must not poll or refill between requests")

    actor._coordinator_step()


def test_sync_request_failure_keeps_coordinator_serving():
    async def run():
        actor = _sync_coordinator_stub()
        actor._refill_exact = lambda count: count

        def fail(_request):
            raise RuntimeError("no materializable trajectories")

        actor._advance_request = fail
        with pytest.raises(RuntimeError, match="no materializable"):
            await asyncio.wait_for(actor.sample(global_steps=1, batch_size=1), timeout=1)
        assert not actor._coordinator_task.done()

        actor._advance_request = lambda _request: (["retry"], {}, 1)
        assert await asyncio.wait_for(actor.sample(global_steps=2, batch_size=1), timeout=1) == (["retry"], {}, 1)
        actor._coordinator_task.cancel()

    asyncio.run(run())


def test_async_sample_waits_for_coordinator_and_refills_after_delivery():
    async def run():
        actor = _async_stub()
        actor.global_steps = 0
        actor.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=2))
        _stub_methods(
            actor,
            poll_interval=0.01,
            poll_once=lambda *_args, **_kwargs: ({"uid"}, {}, True),
            materialize_sampleable=lambda *_args: ["trajectory"],
        )
        actor._pending_sample = None
        actor._wakeup = asyncio.Event()
        actor._coordinator_task = None
        actor._train_state_ready = True
        actor._pending_metrics = {}
        refills = []
        actor._refill_to_watermark = lambda: refills.append(True) or 0

        result = await asyncio.wait_for(actor.sample(1, 1), timeout=1)
        assert result == (["trajectory"], {}, 1)
        assert refills

        actor._coordinator_task.cancel()
        with suppress(asyncio.CancelledError):
            await actor._coordinator_task

    asyncio.run(run())


def test_coordinator_failure_fails_pending_sample_and_terminates_actor(monkeypatch):
    async def run():
        actor = _async_stub()
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            kind=streaming_module._RequestKind.SAMPLE,
            global_steps=1,
            batch_size=1,
            partition_id="train",
            future=future,
        )
        actor._coordinator_task = None

        async def fail():
            raise RuntimeError("coordinator failed")

        actor._coordinator_loop = fail
        exits = []
        monkeypatch.setattr(streaming_module.ray.actor, "exit_actor", lambda: exits.append(True))

        actor._ensure_coordinator()
        with pytest.raises(RuntimeError, match="coordinator failed"):
            await actor._coordinator_task
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="coordinator failed"):
            await future
        assert exits == [True]

    asyncio.run(run())


def test_train_request_uses_one_poll_for_maintenance_and_fulfillment():
    async def run():
        actor = _async_stub()
        actor.global_steps = 3
        actor._pending_sample = None
        actor._pending_metrics = {}
        actor._train_state_ready = True
        polls = []
        refills = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return {"uid-0", "uid-1"}, {}, True

        _stub_methods(
            actor,
            poll_once=poll_once,
            materialize_sampleable=lambda *_args: ["trajectory-0", "trajectory-1"],
        )
        actor._refill_to_watermark = lambda: refills.append("after") or 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            kind=streaming_module._RequestKind.SAMPLE,
            global_steps=3,
            batch_size=2,
            partition_id="train",
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert polls == [(3, "train", 2)]
        assert await future == (["trajectory-0", "trajectory-1"], {}, 2)
        assert refills == ["after"]
        assert actor._pending_sample is None

    asyncio.run(run())


def test_train_eviction_pass_fulfills_request_from_remaining_groups():
    async def run():
        actor = _async_stub()
        actor.global_steps = 3
        actor._pending_metrics = {}
        actor._train_state_ready = True
        eviction = {"training/filter_groups/evicted_samples": 1}
        polls = []
        refills = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return {"uid-0"}, eviction, True

        _stub_methods(
            actor,
            poll_once=poll_once,
            materialize_sampleable=lambda *_args: ["trajectory-0"],
        )
        actor._refill_to_watermark = lambda: refills.append(True) or 1
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            kind=streaming_module._RequestKind.SAMPLE,
            global_steps=3,
            batch_size=1,
            partition_id="train",
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert polls == [(3, "train", 1)]
        assert future.done()
        assert future.result() == (["trajectory-0"], eviction, 1)
        assert refills == [True]
        assert actor._pending_sample is None
        assert actor._pending_metrics == {}

    asyncio.run(run())


def test_train_eviction_refills_without_fulfilling_unready_request():
    async def run():
        actor = _async_stub()
        actor.global_steps = 3
        actor._pending_metrics = {}
        actor._train_state_ready = True
        polls = []
        refills = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return {"uid-0"}, {"training/filter_groups/evicted_samples": 1}, False

        _stub_methods(
            actor,
            poll_once=poll_once,
            materialize_sampleable=lambda *_args: pytest.fail("unready pass must not materialize"),
        )
        actor._refill_to_watermark = lambda: refills.append(True) or 1
        future = asyncio.get_running_loop().create_future()
        request = streaming_module._SampleRequest(
            kind=streaming_module._RequestKind.SAMPLE,
            global_steps=3,
            batch_size=1,
            partition_id="train",
            future=future,
        )
        actor._pending_sample = request

        await _advance_and_maintain_watermark(actor)

        assert polls == [(3, "train", 1)]
        assert refills == [True]
        assert actor._pending_sample is request
        assert not future.done()
        assert actor._pending_metrics == {"training/filter_groups/evicted_samples": 1}
        future.cancel()

    asyncio.run(run())


def test_validation_request_pauses_train_inventory_maintenance():
    async def run():
        actor = _async_stub()
        actor.global_steps = 5
        actor._pending_metrics = {}
        actor._train_state_ready = True
        polls = []
        refills = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return ({"val-uid"} if partition_id == "val" else set()), {}, True

        _stub_methods(
            actor,
            poll_once=poll_once,
            materialize_sampleable=lambda *_args: ["validation-trajectory"],
        )
        actor._refill_to_watermark = lambda: refills.append(True) or 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            kind=streaming_module._RequestKind.SAMPLE,
            global_steps=5,
            batch_size=1,
            partition_id="val",
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert polls == [(5, "val", 1)]
        assert refills == []
        assert await future == (["validation-trajectory"], {}, 1)

    asyncio.run(run())


def test_validation_mode_pauses_periodic_train_maintenance_between_requests():
    async def run():
        actor = _async_stub()
        actor._pending_sample = None
        actor._train_state_ready = True
        actor._validation_active = True
        _stub_methods(actor, poll_once=lambda *_args: pytest.fail("train poll must remain paused during validation"))
        actor._refill_to_watermark = lambda: pytest.fail("train refill must remain paused during validation")

        await _advance_and_maintain_watermark(actor)

    asyncio.run(run())


def test_validation_before_load_does_not_maintain_train_inventory():
    async def run():
        actor = _async_stub()
        actor.global_steps = 0
        actor._pending_metrics = {}
        actor._train_state_ready = False
        polls = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return {"val-uid"}, {}, True

        _stub_methods(
            actor,
            poll_once=poll_once,
            materialize_sampleable=lambda *_args: ["validation-trajectory"],
        )
        actor._refill_to_watermark = lambda: pytest.fail("train refill must remain gated before load")
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            kind=streaming_module._RequestKind.SAMPLE,
            global_steps=0,
            batch_size=1,
            partition_id="val",
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert polls == [(0, "val", 1)]
        assert await future == (["validation-trajectory"], {}, 1)

    asyncio.run(run())


def test_materialize_failure_completes_sample_future_with_exception():
    async def run():
        actor = _async_stub()
        actor._pending_sample = None
        actor._pending_metrics = {}
        actor._train_state_ready = True
        _stub_methods(
            actor,
            poll_once=lambda *_args, **_kwargs: ({"uid"}, {}, True),
            materialize_sampleable=lambda *_args: (_ for _ in ()).throw(RuntimeError("materialize failed")),
        )
        actor._refill_to_watermark = lambda: 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            kind=streaming_module._RequestKind.SAMPLE,
            global_steps=1,
            batch_size=1,
            partition_id="train",
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        with pytest.raises(RuntimeError, match="materialize failed"):
            await future
        assert actor._pending_sample is None

    asyncio.run(run())


def test_non_consuming_count_request_uses_watermark_coordinator():
    async def run():
        actor = _async_stub()
        actor._pending_metrics = {}
        actor._train_state_ready = True
        targets = []
        refills = []

        def poll_once(_step, _partition, target):
            targets.append(target)
            return {"uid-0", "uid-1"}, {}, False

        _stub_methods(actor, poll_once=poll_once)
        actor._refill_to_watermark = lambda: refills.append("after") or 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            kind=streaming_module._RequestKind.COUNT,
            global_steps=1,
            batch_size=0,
            partition_id="train",
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert targets == [0]
        assert await future == (None, {}, 2)
        assert refills == ["after"]

    asyncio.run(run())


def test_non_consuming_wait_request_refills_while_below_target():
    async def run():
        actor = _async_stub()
        actor._pending_metrics = {}
        actor._train_state_ready = True
        targets = []
        refills = []

        def poll_once(_step, _partition, target):
            targets.append(target)
            return {"uid-0"}, {}, False

        _stub_methods(actor, poll_once=poll_once)
        actor._refill_to_watermark = lambda: refills.append(True) or 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            kind=streaming_module._RequestKind.WAIT,
            global_steps=1,
            target_count=2,
            partition_id="train",
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert targets == [2]
        assert refills == [True]
        assert not future.done()
        future.cancel()

    asyncio.run(run())


def _async_rpc_stub(poll_once):
    actor = _async_stub()
    actor.global_steps = 0
    actor._pending_sample = None
    actor._coordinator_task = None
    # Keep idle coordinator passes from consuming scripted polls between requests.
    actor._train_state_ready = False
    actor._pending_metrics = {}
    actor.poll_interval = 0.01
    actor.poll_once = poll_once
    actor._refill_to_watermark = lambda: 0
    return actor


def test_count_sampleable_keeps_eviction_metrics_for_next_request():
    async def run():
        eviction = {"training/rollout_failure/evicted_samples": 1}
        polls = iter([({"uid-0"}, eviction, True), ({"uid-0", "uid-1"}, {}, True)])
        actor = _async_rpc_stub(lambda *_args: next(polls))

        assert await asyncio.wait_for(actor.count_sampleable(3), timeout=1) == 1
        assert actor._pending_metrics == eviction
        assert await asyncio.wait_for(actor.wait_sampleable(3, target_count=2), timeout=1) == eviction
        assert actor._pending_metrics == {}
        actor._coordinator_task.cancel()

    asyncio.run(run())


def test_wait_sampleable_targets_train_partition_without_consuming():
    async def run():
        polls = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return {"uid-0", "uid-1"}, {}, True

        actor = _async_rpc_stub(poll_once)
        actor.materialize_sampleable = lambda *_args: pytest.fail("wait must not consume groups")

        assert await asyncio.wait_for(actor.wait_sampleable(5, target_count=2), timeout=1) == {}
        assert polls == [(5, "train", 2)]
        actor._coordinator_task.cancel()

    asyncio.run(run())


def test_async_checkpoint_rpc_methods_are_synchronous():
    assert not inspect.iscoroutinefunction(StreamingDataloaderAsync.save_checkpoint)
    assert not inspect.iscoroutinefunction(StreamingDataloaderAsync.load_checkpoint)


def test_sync_checkpoint_contains_only_dataloader(monkeypatch, tmp_path):
    actor = _sync_stub()
    actor.train_dataloader = SimpleNamespace(state_dict=lambda: {"cursor": 7})
    tq_saves = []
    monkeypatch.setattr(
        streaming_module.tq,
        "save_checkpoint",
        lambda path, metadata: tq_saves.append((path, metadata)),
        raising=False,
    )

    actor.save_checkpoint(str(tmp_path), global_steps=7)

    assert torch.load(tmp_path / "data.pt", weights_only=False) == {"cursor": 7}
    assert tq_saves == []


def test_async_checkpoint_saves_dataloader_and_delegates_transfer_queue(tmp_path):
    actor = _async_stub()
    actor.train_dataloader = SimpleNamespace(state_dict=lambda: {"cursor": 7})
    tq_saves = []
    actor._save_tq_checkpoint = lambda path, global_steps: tq_saves.append((path, global_steps))

    actor.save_checkpoint(str(tmp_path), global_steps=7)

    assert torch.load(tmp_path / "data.pt", weights_only=False) == {"cursor": 7}
    assert tq_saves == [(str(tmp_path / "transfer_queue"), 7)]


def test_async_restore_loads_dataloader_and_delegates_transfer_queue(tmp_path):
    actor = _async_stub()
    loaded_states = []
    tq_loads = []
    actor.train_dataloader = SimpleNamespace(load_state_dict=loaded_states.append)
    actor._load_tq_checkpoint = lambda path: tq_loads.append(path) or False
    actor._reissue_inflight_prompts = lambda: 0
    actor._refill_to_watermark = lambda: pytest.fail("restore must leave watermark refill to the coordinator")
    actor._ensure_coordinator = lambda: None
    actor._wakeup = SimpleNamespace(set=lambda: None)
    torch.save({"cursor": 5}, tmp_path / "data.pt")

    metrics = actor.load_checkpoint(str(tmp_path), global_steps=5)

    assert loaded_states == [{"cursor": 5}]
    assert tq_loads == [str(tmp_path / "transfer_queue")]
    assert actor.global_steps == 5
    assert actor._train_state_ready is True
    assert metrics == {"streaming_dataloader/reissued_groups": 0.0}


def test_async_restore_requires_dataloader_state(tmp_path):
    actor = _async_stub()
    actor._load_tq_checkpoint = lambda _path: pytest.fail("must not restore TQ alone")

    with pytest.raises(RuntimeError, match="data.pt"):
        actor.load_checkpoint(str(tmp_path), global_steps=5)


def test_submission_registers_prompts_before_dispatch():
    actor = _sync_stub()
    actor.global_steps = 4
    events = []
    actor._register_prompts = lambda batch, partition_id: events.append(("register", partition_id))
    actor.agent_loop_manager = SimpleNamespace(generate_sequences=lambda batch: events.append(("dispatch",)))

    assert actor._submit_batch_to_rollout(["prompt-a", "prompt-b"], "val") == 2
    assert events == [("register", "val"), ("dispatch",)]


def test_trainer_sample_delegates_to_dataloader_actor(monkeypatch):
    calls = []

    class _RemoteSample:
        def remote(self, *args):
            calls.append(args)
            return ("batch", {"metric": 1}, 2)

    stub = SimpleNamespace(streaming_dataloader=SimpleNamespace(sample=_RemoteSample()))
    monkeypatch.setattr(trainer_module.ray, "get", lambda value: value)

    result = PPOTrainer._sample_replay(
        stub,
        global_steps=4,
        partition_id="train",
        batch_size=2,
    )

    assert result == ("batch", {"metric": 1}, 2)
    assert calls == [(4, 2, "train", None)]
