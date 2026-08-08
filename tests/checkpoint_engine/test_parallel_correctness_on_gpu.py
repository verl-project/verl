# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""Correctness test for the parallel-sender NCCL checkpoint engine.

Mirrors test_correctness_on_gpu.py, but with backend="nccl_parallel": one
global bucket sequence round-robined across S = num_trainer senders, received
in ascending global order in bulk-synchronous phases. The rollout workers'
check_weights() compares against the trainer's weights, so a pass certifies
the striped/reordered stream reassembles correctly.

Three rounds per case, because the cross-round state (bound PUB sockets,
connected SUB sockets, end-of-stream markers) is where regressions hide: a
round that leaves an unread marker on a SUB channel corrupts the next one.

The phase < S cases exercise the leftover-marker drain path (groups that are
never read past their last data bucket still hold a queued end-of-stream
marker).

The rebuild_group=True case destroys every group this worker joined (including
the senders' coordination group) in finalize() and re-creates them on the next
round. That is the path the final sender-stream wait exists for: ray's
destroy_collective_group does not synchronize, so a still-in-flight exit
barrier would otherwise be torn down under the collective.

Run (single 8-GPU node):
  python3 -m pytest tests/checkpoint_engine/test_parallel_correctness_on_gpu.py -x -s
"""

import os

import pytest
import ray
import torch

from tests.checkpoint_engine.test_utils import create_rollout_worker_group, create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray.base import (
    RayResourcePool,
    split_resource_pool,
)
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig

_ngpus = torch.cuda.device_count()


@pytest.mark.skipif(_ngpus < 8, reason="the phase < S cases need 4 trainer + 4 rollout workers")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "num_trainer, num_rollout, phase, rebuild_group",
    [
        (1, _ngpus - 1, None, False),  # S=1: degenerates to a single-sender engine
        (2, _ngpus - 2, None, False),  # phase defaults to S
        (4, _ngpus - 4, None, False),
        (4, _ngpus - 4, 3, False),  # non-divisor phase < S: partial final phase + marker drain
        (4, _ngpus - 4, 2, False),  # phase < S: leftover-marker drain
        (2, _ngpus - 2, None, True),  # groups destroyed and rebuilt between rounds
    ],
)
@auto_await
async def test_nccl_parallel_checkpoint_engine(
    num_trainer,
    num_rollout,
    phase,
    rebuild_group,
    num_nodes=1,
    num_gpus_per_node=_ngpus,
    bucket_size_mb=128,
    check_allclose=True,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    model_path = os.path.expanduser(model_path)
    ray.init(
        runtime_env={
            "env_vars": {
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    try:
        parallel_kwargs = {"rebuild_group": rebuild_group}
        if phase is not None:
            parallel_kwargs["phase"] = phase
        checkpoint_engine_config = CheckpointEngineConfig(
            backend="nccl_parallel",
            update_weights_bucket_megabytes=bucket_size_mb,
            engine_kwargs={"nccl_parallel": parallel_kwargs},
        )
        model_config = HFModelConfig(path=model_path, use_remove_padding=True)
        # TP=1 so every rollout worker forms its own replica: the default TP floors
        # odd worker counts into orphaned workers that receive nothing.
        rollout_config = RolloutConfig(
            name="vllm", checkpoint_engine=checkpoint_engine_config, tensor_model_parallel_size=1
        )

        resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
        trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
        actor_wg = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
        actor_wg.reset()
        rollout, replicas = await create_rollout_worker_group(
            rollout_pool, model_config, rollout_config, check_allclose
        )

        # Every rollout worker must belong to a replica, or it silently receives
        # nothing and the check reports it as missing weights.
        assert len(replicas) * rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size == (
            rollout.world_size
        ), f"orphaned rollout workers: {rollout.world_size} workers, {len(replicas)} replicas"

        checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=actor_wg, replicas=replicas
        )
        for _ in range(3):
            await checkpoint_manager.update_weights()
            rollout.check_weights()
    finally:
        # Always release the local Ray instance -- covers failures anywhere
        # after ray.init(), or every later param case dies on double init.
        ray.shutdown()


def test_invalid_phase_rejected():
    """phase outside [1, S] must raise, not silently clamp."""
    eng = pytest.importorskip("verl.checkpoint_engine.nccl_parallel_checkpoint_engine")

    # Order sensitivity: the ctor applies the process-wide non-blocking
    # stream-factory patch and fails closed if a driver-side ray stream pool
    # already exists in this pytest process (no current test creates one).
    engine = eng.NCCLParallelCheckpointEngine(bucket_size=256 << 20, phase=0)
    meta = [type("M", (), {"zmq_ip": "127.0.0.1", "zmq_port": 6000 + i})() for i in range(4)]
    with pytest.raises(ValueError, match=r"phase must be an int"):
        engine.init_process_group([0, -1, -1, -1], 5, meta)


def test_bucket_size_alignment():
    """The per-group bucket must land on a 16-byte boundary for every S.

    2048MB / 24 senders is odd; an unaligned bucket size lets
    split_weight_chunks cut a large tensor so that every later tensor in the
    bucket sits at a misaligned offset, and the receive path's dtype view
    raises (``storage_offset() must be divisible by 2 to view Byte as
    BFloat16``). The phase=0 ValueError fires right after the bucket-size
    computation, before any CUDA/NCCL work, so this probes the real code path
    without a GPU.

    Two budgets, because a whole-megabyte budget is a multiple of 2**20 and so
    divides evenly by every power-of-two sender count: at 2048MB only S=3/7/24
    would fail without the alignment step. The byte-granular budget (2048MB
    minus one byte) leaves S=1/8/16 unaligned too, so every S above the floor
    exercises the alignment. S=192 is the floor probe in both cases: the budget
    per sender lands under the 32MB floor, which is itself aligned.
    """
    eng = pytest.importorskip("verl.checkpoint_engine.nccl_parallel_checkpoint_engine")

    for total_bucket in ((2048 << 20), (2048 << 20) - 1):
        for num_senders in (1, 3, 7, 8, 16, 24, 24 * 8):
            engine = eng.NCCLParallelCheckpointEngine(bucket_size=total_bucket, phase=0)
            meta = [type("M", (), {"zmq_ip": "127.0.0.1", "zmq_port": 6000 + i})() for i in range(num_senders)]
            with pytest.raises(ValueError, match=r"phase must be an int"):
                engine.init_process_group([0] + [-1] * (num_senders - 1), num_senders + 1, meta)
            assert engine.bucket_size % 16 == 0, (
                f"budget={total_bucket} S={num_senders}: bucket {engine.bucket_size} not 16-byte aligned"
            )
            assert engine.bucket_size >= 32 << 20, (
                f"budget={total_bucket} S={num_senders}: bucket below the documented floor"
            )
