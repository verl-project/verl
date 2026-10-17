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

"""Real two-worker budget enforcement and deterministic cancellation races."""

import asyncio
import os
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from verl import DataProto
from verl.experimental.reward_loop.reward_budget import (
    RewardConcurrencyBudget,
    create_reward_budget,
    run_concurrency_limit,
    shared_reward_slot,
)
from verl.experimental.reward_loop.reward_loop import RewardLoopManager
from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager


def _data(count=1):
    return DataProto.from_dict(
        tensors={
            "responses": torch.ones(count, 1, dtype=torch.long),
            "attention_mask": torch.ones(count, 2, dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array(["test"] * count),
            "reward_model": np.array([{"ground_truth": "answer"} for _ in range(count)], dtype=object),
            "extra_info": np.array([{"sample": i} for i in range(count)], dtype=object),
        },
    )


class _LocalRPC:
    """Controllable transport for race tests; distributed enforcement is tested below."""

    def __init__(self, state):
        async def acquire(token):
            return state.try_acquire(token)

        async def release(token):
            state.release(token)

        self.try_acquire = SimpleNamespace(remote=acquire)
        self.release = SimpleNamespace(remote=release)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2"])
def test_invalid_limits_are_rejected(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        run_concurrency_limit({"run_max_concurrent": invalid})


def test_budget_is_idempotent_and_independent():
    first, second = RewardConcurrencyBudget(1), RewardConcurrencyBudget(1)
    assert first.try_acquire("a")
    assert first.try_acquire("a")
    assert not first.try_acquire("b")
    assert second.try_acquire("b")
    first.release("unknown")
    assert not first.try_acquire("b")
    first.release("a")
    first.release("a")
    assert first.try_acquire("b")


@pytest.mark.parametrize("phase", ["acquire", "waiting", "scoring", "release"])
def test_cancellation_does_not_overtake_pending_grant_or_release(phase):
    async def scenario():
        state = RewardConcurrencyBudget(1)
        rpc = _LocalRPC(state)
        entered, continue_rpc = asyncio.Event(), asyncio.Event()
        original_acquire, original_release = rpc.try_acquire.remote, rpc.release.remote
        if phase == "waiting":
            assert state.try_acquire("occupied")

        async def acquire(token):
            if phase in ("acquire", "waiting"):
                entered.set()
            if phase == "acquire":
                await continue_rpc.wait()
            return await original_acquire(token)

        async def release(token):
            if phase == "release":
                entered.set()
                await continue_rpc.wait()
            await original_release(token)

        rpc.try_acquire.remote, rpc.release.remote = acquire, release

        async def run():
            async with shared_reward_slot(rpc):
                if phase == "scoring":
                    entered.set()
                    await asyncio.Event().wait()

        task = asyncio.create_task(run())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        if phase in ("acquire", "release"):
            assert not task.done()
            task.cancel()  # Repeated cancellation must not abandon cleanup.
            continue_rpc.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert state.active == ({"occupied"} if phase == "waiting" else set())

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["ok", "error", "timeout", "cancel", "cancel_twice"])
def test_scorer_cleanup_finishes_before_releasing_capacity(monkeypatch, outcome):
    async def scenario():
        monkeypatch.setattr(RateLimitedRewardManager, "_class_initialized", False)
        state = RewardConcurrencyBudget(1)
        entered, cleanup_started, finish_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def score(**kwargs):
            entered.set()
            try:
                if outcome in ("timeout", "cancel", "cancel_twice"):
                    await asyncio.Event().wait()
                if outcome == "error":
                    raise RuntimeError("judge failed")
                return 1.0
            finally:
                cleanup_started.set()
                await finish_cleanup.wait()

        manager = RateLimitedRewardManager(
            OmegaConf.create({"reward": {"run_max_concurrent": 1, "timeout": 0.1 if outcome == "timeout" else 30}}),
            SimpleNamespace(decode=lambda *a, **k: "answer"),
            score,
            reward_budget=_LocalRPC(state),
        )
        task = asyncio.create_task(manager.run_single(_data()))
        await asyncio.wait_for(entered.wait(), 2)
        if outcome.startswith("cancel"):
            task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 2)
        if outcome == "cancel_twice":
            task.cancel()
            for _ in range(10):
                await asyncio.sleep(0)
            assert not task.done(), "caller must wait for scorer cleanup even after repeated cancellation"
        assert len(state.active) == 1
        assert not state.try_acquire("other")
        finish_cleanup.set()
        if outcome.startswith("cancel"):
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result["reward_score"] == (1.0 if outcome == "ok" else 0.0)
            if outcome != "ok":
                assert result["reward_extra_info"][outcome]
        assert not state.active

    asyncio.run(scenario())


def test_default_is_unchanged_and_unsupported_config_is_rejected(monkeypatch):
    monkeypatch.setattr(ray, "remote", lambda *a, **k: pytest.fail("must not create coordinator"))
    assert create_reward_budget(OmegaConf.create({"reward": {}})) is None
    config = OmegaConf.create(
        {"reward": {"run_max_concurrent": 2, "reward_manager": {"source": "register", "name": "naive"}}}
    )
    with pytest.raises(ValueError, match="registered rate_limited"):
        create_reward_budget(config)
    config.reward.reward_manager.name = "rate_limited"
    with pytest.raises(ValueError, match="custom async"):
        create_reward_budget(config)
    with pytest.raises(ValueError, match="Manager-owned"):
        RateLimitedRewardManager(config, SimpleNamespace(), lambda **kwargs: 1.0)
    with pytest.raises(ValueError, match="custom async"):
        RateLimitedRewardManager(config, SimpleNamespace(), lambda **kwargs: 1.0, reward_budget=object())


def _observer_class():
    # A local class is serialized by value; workers need not import the pytest module.
    class Observer:
        def __init__(self):
            self.active = 0
            self.peak = 0
            self.pids = set()
            self.gate = asyncio.Event()

        async def enter(self, pid):
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.pids.add(pid)

        async def leave(self):
            self.active -= 1

        async def wait(self):
            await self.gate.wait()

        async def release(self):
            self.gate.set()

        async def snapshot(self):
            return self.active, self.peak, self.pids

    return Observer


@pytest.fixture
def ray_runtime():
    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=128 * 1024 * 1024, logging_level="ERROR")
    try:
        yield
    finally:
        ray.shutdown()


@pytest.mark.parametrize(("run_limit", "is_async"), [(None, True), (2, True), (2, False)])
def test_real_workers_load_scorer_and_enforce_one_run_budget(tmp_path, ray_runtime, run_limit, is_async):
    """Load worker-only scorers in workers, rejecting sync scorers with a shared budget."""
    tokenizer_path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"answer": 1, "[UNK]": 0}, unk_token="[UNK]")), unk_token="[UNK]"
    ).save_pretrained(tokenizer_path)
    observer_name = "reward-budget-observer-" + uuid.uuid4().hex
    observer = ray.remote(num_cpus=0)(_observer_class()).options(name=observer_name).remote()
    scorer = tmp_path / "async_score.py"
    scorer.write_text(
        "import os\nimport ray\n"
        f"if os.getpid() == {os.getpid()}:\n"
        "    raise RuntimeError('scorer must only be imported in a reward worker')\n"
        "async def compute_score(observer_name, **kwargs):\n"
        "    observer = ray.get_actor(observer_name)\n"
        "    await observer.enter.remote(os.getpid())\n"
        "    try:\n"
        "        await observer.wait.remote()\n"
        "        return 1.0\n"
        "    finally:\n"
        "        await observer.leave.remote()\n"
    )
    if not is_async:
        with scorer.open("a") as stream:
            stream.write("def compute_score(observer_name, **kwargs):\n    return 1.0\n")
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {"tokenizer_path": str(tokenizer_path), "path": str(tokenizer_path)}},
            "reward": {
                "num_workers": 2,
                "max_concurrent": 2,
                "run_max_concurrent": run_limit,
                "timeout": 60,
                "reward_model": {"enable": False},
                "reward_manager": {"source": "register", "name": "rate_limited"},
                "custom_reward_function": {
                    "path": str(scorer),
                    "name": "compute_score",
                    "reward_kwargs": {"observer_name": observer_name},
                },
            },
        }
    )
    manager = RewardLoopManager(config)
    if not is_async:
        for worker in manager.reward_loop_workers:
            with pytest.raises(ray.exceptions.RayActorError, match="custom async reward function"):
                ray.get(worker.__ray_ready__.remote(), timeout=30)
        return
    pending = [worker.compute_score_batch.remote(_data(2)) for worker in manager.reward_loop_workers]
    expected_peak = 4 if run_limit is None else run_limit
    deadline = time.monotonic() + 60
    while True:
        active, peak, pids = ray.get(observer.snapshot.remote(), timeout=5)
        if active >= expected_peak:
            break
        assert time.monotonic() < deadline, (active, peak, pids)
        time.sleep(0.05)
    ray.get(observer.release.remote(), timeout=5)
    results = ray.get(pending, timeout=30)
    assert [item["reward_score"] for batch in results for item in batch] == [1.0] * 4
    active, peak, pids = ray.get(observer.snapshot.remote(), timeout=5)
    assert active == 0
    assert peak == expected_peak
    assert len(pids) == 2 and os.getpid() not in pids


def test_dead_coordinator_fails_instead_of_becoming_zero_reward(ray_runtime, monkeypatch):
    async def scenario():
        monkeypatch.setattr(RateLimitedRewardManager, "_class_initialized", False)
        budget = ray.remote(num_cpus=0, max_restarts=0)(RewardConcurrencyBudget).remote(1)
        assert await budget.try_acquire.remote("probe")
        ray.kill(budget, no_restart=True)

        async def score(**kwargs):
            pytest.fail("a dead coordinator must not admit scorer work")

        manager = RateLimitedRewardManager(
            OmegaConf.create({"reward": {"run_max_concurrent": 1}}),
            SimpleNamespace(decode=lambda *a, **k: "answer"),
            score,
            reward_budget=budget,
        )
        with pytest.raises(ray.exceptions.RayActorError):
            await asyncio.wait_for(manager.run_single(_data()), 10)

    asyncio.run(scenario())


def test_admission_timeout_is_not_a_zero_reward(monkeypatch):
    async def scenario():
        monkeypatch.setattr(RateLimitedRewardManager, "_class_initialized", False)
        state = RewardConcurrencyBudget(1)
        assert state.try_acquire("busy")

        async def score(**kwargs):
            pytest.fail("unadmitted scorer must not start")

        manager = RateLimitedRewardManager(
            OmegaConf.create({"reward": {"run_max_concurrent": 1, "timeout": 0.05}}),
            SimpleNamespace(decode=lambda *a, **k: "answer"),
            score,
            reward_budget=_LocalRPC(state),
        )
        with pytest.raises(TimeoutError, match="shared reward budget"):
            await asyncio.wait_for(manager.run_single(_data()), 2)
        assert state.active == {"busy"}

    asyncio.run(scenario())


def test_worker_death_does_not_reissue_its_capacity(ray_runtime):
    class Worker:
        async def hold(self, budget):
            return await budget.try_acquire.remote("dead-worker")

    budget = ray.remote(num_cpus=0, max_restarts=0)(RewardConcurrencyBudget).remote(1)
    worker = ray.remote(num_cpus=0, max_restarts=0)(Worker).remote()
    assert ray.get(worker.hold.remote(budget), timeout=10)
    ray.kill(worker, no_restart=True)

    async def scenario():
        with pytest.raises(TimeoutError, match="Restart the job"):
            async with shared_reward_slot(budget, timeout=0.1):
                pytest.fail("a dead worker's external work may still be running")
        assert not await budget.try_acquire.remote("replacement")

    asyncio.run(scenario())
