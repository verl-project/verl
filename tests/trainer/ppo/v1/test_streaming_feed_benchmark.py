# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Manual benchmark for streaming prompt submission without running training."""

import os

import pytest

if os.getenv("RUN_STREAMING_FEED_BENCHMARK") != "1":
    pytest.skip("Set RUN_STREAMING_FEED_BENCHMARK=1 to run the manual benchmark.", allow_module_level=True)

import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import numpy as np  # noqa: E402
import ray  # noqa: E402
import transfer_queue as tq  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from verl.trainer.ppo.v1 import trainer_base  # noqa: E402
from verl.trainer.ppo.v1.trainer_base import PPOTrainer  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402


@ray.remote
class _NoopAgentLoopWorker:
    def __init__(self):
        self.calls = 0
        self.prompts = 0

    def generate_sequences(self, batch):
        self.calls += 1
        self.prompts += len(batch)

    def reset(self):
        self.calls = 0
        self.prompts = 0

    def stats(self):
        return {"calls": self.calls, "prompts": self.prompts}


class _NoopAgentLoopManager:
    """Match AgentLoopManagerTQ dispatch while skipping rollout."""

    def __init__(self, num_workers: int):
        self.workers = [_NoopAgentLoopWorker.remote() for _ in range(num_workers)]

    def generate_sequences(self, prompts):
        chunks = prompts.chunk(len(self.workers))
        ray.get([worker.generate_sequences.remote(chunk) for worker, chunk in zip(self.workers, chunks, strict=False)])

    def reset(self):
        ray.get([worker.reset.remote() for worker in self.workers])

    def stats(self):
        return ray.get([worker.stats.remote() for worker in self.workers])

    def close(self):
        for worker in self.workers:
            ray.kill(worker)


class _SyntheticLoader:
    def __init__(self, batch_size: int):
        self.batch = {
            "raw_prompt": [[{"role": "user", "content": "streaming feed benchmark"}] for _ in range(batch_size)],
            "index": np.arange(batch_size, dtype=np.int64),
        }

    def __iter__(self):
        return self

    def __next__(self):
        return dict(self.batch)


class _BenchmarkTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


@pytest.fixture(scope="module")
def benchmark_runtime():
    started_ray = not ray.is_initialized()
    if started_ray:
        address = os.getenv("STREAMING_FEED_BENCHMARK_RAY_ADDRESS")
        ray.init(**({"address": address} if address else {}))
    tq.init()
    yield
    tq.close()
    if started_ray:
        ray.shutdown()


def _make_trainer(gen_batch_size: int, train_batch_size: int, manager: _NoopAgentLoopManager):
    trainer = _BenchmarkTrainer.__new__(_BenchmarkTrainer)
    trainer.config = OmegaConf.create(
        {"data": {"train_batch_size": train_batch_size, "gen_batch_size": gen_batch_size}}
    )
    trainer.train_dataloader = _SyntheticLoader(gen_batch_size)
    trainer.train_dataloader_it = None
    trainer.global_steps = 1
    trainer.trainer_mode = "separate_async"
    trainer.agent_loop_manager = manager
    return trainer


def _clear_partition(partition_id: str):
    partition = tq.kv_list(partition_id=partition_id) or {}
    keys = list(partition.get(partition_id, {}))
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def _add_batch_to_generate_fragmented(trainer: _BenchmarkTrainer, num_prompts: int | None = None) -> int:
    """Previous behavior: submit every dataloader chunk separately."""
    train_batch_size = trainer.config.data.train_batch_size
    if num_prompts is None:
        num_prompts = train_batch_size
    gen_batch_size = trainer.config.data.get("gen_batch_size", None) or train_batch_size
    if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
        raise ValueError("num_prompts must be a positive multiple of gen_batch_size")

    submitted = 0
    while submitted < num_prompts:
        batch = trainer._fetch_one_gen_batch()
        tu.assign_non_tensor_data(batch, "global_steps", trainer.global_steps)
        tags = [{"is_prompt": True, "status": "pending", "global_steps": trainer.global_steps}] * len(batch)
        tq.kv_batch_put(
            keys=list(batch["uid"]),
            partition_id="train",
            tags=tags,
            fields=trainer._storable_prompt_fields(batch),
        )
        trainer.agent_loop_manager.generate_sequences(batch)
        submitted += len(batch)
    return submitted


def _measure(
    *,
    gen_batch_size: int,
    train_batch_size: int,
    fragmented: bool,
    warmup: int,
    repeats: int,
    manager: _NoopAgentLoopManager,
    partition_id: str,
):
    trainer = _make_trainer(gen_batch_size, train_batch_size, manager)
    submit = _add_batch_to_generate_fragmented if fragmented else _BenchmarkTrainer._add_batch_to_generate

    for _ in range(warmup):
        assert submit(trainer) == train_batch_size
        _clear_partition(partition_id)

    manager.reset()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        assert submit(trainer) == train_batch_size
        durations.append(time.perf_counter() - start)
        _clear_partition(partition_id)

    return durations, manager.stats()


def _percentile(samples: list[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(samples)) - 1)
    return sorted(samples)[index]


def test_compare_fragmented_coalesced_and_full_batch(benchmark_runtime, monkeypatch):
    train_batch_size = int(os.getenv("STREAMING_FEED_BENCHMARK_TRAIN_BATCH_SIZE", "64"))
    num_workers = int(os.getenv("STREAMING_FEED_BENCHMARK_WORKERS", "4"))
    warmup = int(os.getenv("STREAMING_FEED_BENCHMARK_WARMUP", "2"))
    repeats = int(os.getenv("STREAMING_FEED_BENCHMARK_REPEATS", "10"))
    assert train_batch_size > 1 and num_workers > 0 and warmup >= 0 and repeats > 0

    partition_id = f"streaming-feed-benchmark-{uuid.uuid4().hex}"
    original_kv_batch_put = tq.kv_batch_put

    def benchmark_kv_batch_put(*args, **kwargs):
        kwargs["partition_id"] = partition_id
        return original_kv_batch_put(*args, **kwargs)

    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", benchmark_kv_batch_put)
    manager = _NoopAgentLoopManager(num_workers)

    try:
        results = {}
        variants = [
            ("fragmented gen_batch_size=1", 1, True),
            ("coalesced gen_batch_size=1", 1, False),
            (f"gen_batch_size={train_batch_size}", train_batch_size, False),
        ]
        for label, gen_batch_size, fragmented in variants:
            durations, worker_stats = _measure(
                gen_batch_size=gen_batch_size,
                train_batch_size=train_batch_size,
                fragmented=fragmented,
                warmup=warmup,
                repeats=repeats,
                manager=manager,
                partition_id=partition_id,
            )
            results[label] = durations
            calls_per_trial = (
                (train_batch_size // gen_batch_size) * min(gen_batch_size, num_workers)
                if fragmented
                else min(train_batch_size, num_workers)
            )
            expected_calls = repeats * calls_per_trial
            assert sum(stat["calls"] for stat in worker_stats) == expected_calls
            assert sum(stat["prompts"] for stat in worker_stats) == repeats * train_batch_size
            print(
                f"\n{label}: "
                f"median={statistics.median(durations) * 1000:.2f} ms, "
                f"p95={_percentile(durations, 0.95) * 1000:.2f} ms, "
                f"worker_prompts={[stat['prompts'] for stat in worker_stats]}"
            )

        fragmented = statistics.median(results["fragmented gen_batch_size=1"])
        coalesced = statistics.median(results["coalesced gen_batch_size=1"])
        full_batch = statistics.median(results[f"gen_batch_size={train_batch_size}"])
        print(f"fragmented / coalesced median ratio = {fragmented / coalesced:.2f}x")
        print(f"coalesced / full-batch median ratio = {coalesced / full_batch:.2f}x")
    finally:
        _clear_partition(partition_id)
        manager.close()
