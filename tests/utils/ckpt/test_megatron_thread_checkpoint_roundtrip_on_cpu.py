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

"""Real MCore writes and loads, with only its CUDA-only control flags adapted to CPU."""

import copy
import gc
import threading
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from verl.utils.checkpoint.megatron_thread_checkpoint import ThreadedCheckpointWriter, create_thread_checkpoint_queue


def _check_rank_local_failures(rank, rendezvous):
    from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
    from torch.distributed.checkpoint import CheckpointException

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    with pytest.MonkeyPatch.context() as patch:
        # Adapt the native queue's CUDA control scalar to this CPU process group.
        patch.setattr(torch.cuda, "current_device", lambda: "cpu")
        failure_phase = None
        writes, finalized = [], []

        def preload():
            if rank == 1 and failure_phase == "preload":
                raise RuntimeError("rank 1 failed during preload")
            return []

        def synchronize():
            if rank == 1 and failure_phase == "fence":
                raise RuntimeError("rank 1 failed during fence")

        queue = create_thread_checkpoint_queue(synchronize=synchronize)
        writer = ThreadedCheckpointWriter(queue=queue)
        request = AsyncRequest(lambda *_args: writes.append(rank), (rank, [], None), [], preload_fn=preload)
        submitted = []
        native_submit = queue._executor.submit

        def submit(*args, **kwargs):
            future = native_submit(*args, **kwargs)
            submitted.append(future)
            # An executor can enqueue its work item before thread startup fails.
            if rank == 1 and failure_phase == "submit":
                raise RuntimeError("rank 1 failed during submit")
            return future

        patch.setattr(queue._executor, "submit", submit)
        try:
            for failure_phase in ("preload", "fence", "submit"):
                with pytest.raises(CheckpointException, match=f"rank 1 failed during {failure_phase}") as captured:
                    writer.schedule([request], lambda: finalized.append(rank))

                assert set(captured.value.failures) == {1}
                assert queue.get_num_unfinalized_calls() == 0
                assert not writer.pending
                assert writer.finalize()
                for future in submitted:
                    future.result(timeout=5)
                assert not writes and not finalized

            failure_phase = None
            writer.schedule([request], lambda: finalized.append(rank))
            assert writer.pending
            assert writer.finalize()
            assert writes == [rank] and finalized == [rank]

            def finalize_manifest():
                if rank == 0:
                    raise OSError("rank 0 failed to publish the role manifest")

            writer.schedule([request], finalize_manifest)
            with pytest.raises(CheckpointException, match="rank 0 failed to publish the role manifest") as captured:
                writer.finalize()
            assert set(captured.value.failures) == {0}
            assert writer.pending and not writer.finalize()
            assert writes == [rank, rank]
        finally:
            writer.close()
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for two-rank checkpoint tests")
def test_rank_local_scheduling_and_manifest_failures_reach_every_rank(tmp_path):
    pytest.importorskip("megatron.core", minversion="0.19.2")
    torch.multiprocessing.spawn(_check_rank_local_failures, args=(f"file://{tmp_path / 'gloo'}",), nprocs=2)


@pytest.fixture
def cpu_native_runtime(monkeypatch, tmp_path):
    """Keep real Gloo collectives, native planning, serialization and finalizers."""
    pytest.importorskip("megatron.core", minversion="0.19.2")
    from megatron.core import mpu
    from megatron.core.dist_checkpointing.strategies import filesystem_async, state_dict_saver

    assert not torch.cuda.is_initialized()
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'gloo'}", rank=0, world_size=1)
    assert dist.get_world_size() == 1 and dist.get_backend() == "gloo"
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda **kwargs: dist.group.WORLD)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    # Seed the legacy channel so native planning need not spawn a Manager.
    # Threaded requests must neither replace it nor consume its results.
    legacy_results = Queue()
    legacy_results.put("legacy-result")
    monkeypatch.setattr(filesystem_async, "_results_queue", legacy_results)

    def cpu_collective_flag(data, *args, **kwargs):
        # MCore 0.19.2's cached-plan agreement uses a literal CUDA device.
        # Redirect only that one-element integer control tensor, not payloads.
        if kwargs.get("device") == "cuda":
            assert kwargs.get("dtype") == torch.int and torch.as_tensor(data).numel() == 1
            kwargs["device"] = "cpu"
        return torch.tensor(data, *args, **kwargs)

    monkeypatch.setattr(
        state_dict_saver,
        "torch",
        SimpleNamespace(tensor=cpu_collective_flag, int=torch.int, distributed=dist, cuda=torch.cuda),
    )
    try:
        yield SimpleNamespace(legacy_results=legacy_results, filesystem_async=filesystem_async)
        assert not torch.cuda.is_initialized()
        assert filesystem_async._results_queue is legacy_results
        assert legacy_results.get_nowait() == "legacy-result" and legacy_results.empty()
    finally:
        if owns_group:
            dist.destroy_process_group()


def _update(parameter, optimizer, scheduler, step):
    parameter.grad = torch.arange(1, parameter.numel() + 1, dtype=parameter.dtype) * (step / 100)
    optimizer.step()
    scheduler.step()


def _shard(key, tensor):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    return ShardedTensor.from_rank_offsets(key, tensor)


def _load_trees(root, expected):
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy

    loaded = {}
    for content in ("model", "optimizer", "extra"):
        template = {
            key: _shard(key, torch.empty_like(value))
            for key, value in expected.items()
            if key.startswith(content + ".")
        }
        loaded.update(
            dist_checkpointing.load(
                template,
                root / content / "dist_ckpt",
                sharded_strategy=FullyParallelLoadStrategyWrapper(TorchDistLoadShardedStrategy(), dist.group.WORLD),
            )
        )
    return loaded


@pytest.mark.parametrize("cache_plans", [False, True])
def test_threaded_native_three_tree_roundtrip_reuses_writer_and_restores_next_adam_update(
    cpu_native_runtime, tmp_path, cache_plans
):
    parameter = torch.nn.Parameter(torch.arange(16, dtype=torch.float32) / 16)
    parameter.data.share_memory_()
    optimizer = torch.optim.AdamW([parameter], lr=0.03, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    queue = create_thread_checkpoint_queue(synchronize=lambda: None, sync_all_done=lambda alive: not alive)
    writer = ThreadedCheckpointWriter(cache_plans=cache_plans, queue=queue)
    finalized = []
    actor_thread = threading.get_ident()
    io_threads = []
    result_channels = []
    gc_enabled = gc.isenabled()
    try:
        for step in (1, 2):
            _update(parameter, optimizer, scheduler, step * 2 - 1)
            live_tensors = {"model.weight": parameter.detach()}
            for name, value in optimizer.state[parameter].items():
                value.share_memory_()
                live_tensors["optimizer." + name] = value
            expected = {key: value.clone() for key, value in live_tensors.items()}
            extra = {
                "step": step,
                "scheduler": copy.deepcopy(scheduler.state_dict()),
                "optimizer_param_groups": copy.deepcopy(optimizer.state_dict()["param_groups"]),
                "rng": torch.get_rng_state(),
            }
            root = tmp_path / f"step_{step}"
            requests = []
            for content in ("model", "optimizer", "extra"):
                state = {
                    key: _shard(key, value) for key, value in live_tensors.items() if key.startswith(content + ".")
                }
                if content == "extra":
                    state = {"extra": extra}
                path = root / content / "dist_ckpt"
                path.mkdir(parents=True)
                requests.append(writer.save_dist_checkpoint(path, state))

            channels = [request.async_fn_args[2] for request in requests if request.async_fn is not None]
            assert len(channels) >= 2 and all(channel is not cpu_native_runtime.legacy_results for channel in channels)
            result_channels.extend(channels)
            assert len({id(channel) for channel in result_channels}) == len(result_channels)
            assert cpu_native_runtime.filesystem_async._results_queue is cpu_native_runtime.legacy_results

            # Hold the first native write until the real optimizer mutation.
            # The wrapper invokes the unmodified MCore writer after release.
            writing, release = threading.Event(), threading.Event()
            native_write = requests[0].async_fn

            def gated_native_write(*args, _writing=writing, _release=release, _native_write=native_write, **kwargs):
                io_threads.append(threading.get_ident())
                _writing.set()
                assert _release.wait(5)
                return _native_write(*args, **kwargs)

            requests[0] = requests[0]._replace(async_fn=gated_native_write)
            try:
                metrics = writer.schedule(requests, lambda step=step: finalized.append(step))
                assert writing.wait(5)
                assert not writer.finalize(blocking=False, no_dist=True)
                assert finalized == list(range(1, step))
                assert metrics["checkpoint/worker_snapshot_bytes"] == sum(
                    t.numel() * t.element_size() for t in expected.values()
                )
                assert (
                    metrics["checkpoint/worker_cpu_snapshot_copy_bytes"] == metrics["checkpoint/worker_snapshot_bytes"]
                )
                _update(parameter, optimizer, scheduler, step * 2)
            finally:
                release.set()
            assert writer.finalize(blocking=True)
            assert finalized == list(range(1, step + 1)) and gc.isenabled() is gc_enabled

            loaded = _load_trees(root, expected)
            for key, value in expected.items():
                torch.testing.assert_close(loaded[key], value, atol=0, rtol=0)
            assert loaded["extra"]["scheduler"] == extra["scheduler"]
            assert loaded["extra"]["step"] == step
            torch.testing.assert_close(loaded["extra"]["rng"], extra["rng"], atol=0, rtol=0)

            restored_parameter = torch.nn.Parameter(loaded["model.weight"])
            restored_optimizer = torch.optim.AdamW([restored_parameter], foreach=False)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=2, gamma=0.8)
            restored_optimizer.load_state_dict(
                {
                    "state": {0: {name: loaded["optimizer." + name] for name in optimizer.state[parameter]}},
                    "param_groups": loaded["extra"]["optimizer_param_groups"],
                }
            )
            restored_scheduler.load_state_dict(loaded["extra"]["scheduler"])
            _update(restored_parameter, restored_optimizer, restored_scheduler, step * 2)
            torch.testing.assert_close(restored_parameter, parameter, atol=0, rtol=0)
            for name, value in optimizer.state[parameter].items():
                torch.testing.assert_close(restored_optimizer.state[restored_parameter][name], value, atol=0, rtol=0)
            assert restored_scheduler.state_dict() == scheduler.state_dict()
        assert len(io_threads) == 2 and len(set(io_threads)) == 1 and io_threads[0] != actor_thread
    finally:
        writer.close()
