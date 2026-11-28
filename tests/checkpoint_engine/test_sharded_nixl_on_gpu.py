# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""GPU tests of the ``sharded_nixl`` checkpoint engine: real FSDP/Megatron trainers send to mock
rollout workers over NIXL, which compare every received weight bit-exactly with the checkpoint.

Requires NIXL + UCX; set ``UCX_NET_DEVICES`` / ``UCX_TLS`` for your fabric.

Run (one node):   pytest -s tests/checkpoint_engine/test_sharded_nixl_on_gpu.py
Run (two nodes):  start a Ray cluster over both nodes, then
                  NUM_NODES=2 pytest -s tests/checkpoint_engine/test_sharded_nixl_on_gpu.py -k "2-2"
"""

import os

import pytest
import ray
import torch
from transformers import AutoConfig

from tests.checkpoint_engine.test_utils import create_rollout_worker_group, create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig

pytest.importorskip("nixl")

_ngpus = torch.cuda.device_count()
_num_nodes = int(os.environ.get("NUM_NODES", "1"))
_model = os.environ.get("SHARDED_NIXL_TEST_MODEL", "~/models/Qwen/Qwen3-0.6B")
_moe_model = os.environ.get("SHARDED_NIXL_MOE_TEST_MODEL", "~/models/Qwen/Qwen3-30B-A3B")

_UCX_ENV = {
    "UCX_TLS": os.environ.get("UCX_TLS", "rc,cuda_copy,cuda_ipc,tcp"),
    "UCX_MAX_RNDV_RAILS": os.environ.get("UCX_MAX_RNDV_RAILS", "4"),
    "UCX_LOG_LEVEL": os.environ.get("UCX_LOG_LEVEL", "WARN"),
    "VERL_LOGGING_LEVEL": "DEBUG",
}
for _k in ("UCX_NET_DEVICES", "UCX_IB_GID_INDEX", "UCX_IB_ROCE_REACHABILITY_MODE"):
    if _k in os.environ:
        _UCX_ENV[_k] = os.environ[_k]


async def _run(
    backend_kwargs: dict,
    num_trainer: int,
    num_rollout: int,
    bucket_size_mb: int = 128,
    rollout_layouts=None,
    expect_error: type | None = None,
    num_syncs: int = 3,
    rollout_backend_kwargs: dict | None = None,
    rollout_tp: int = 1,
    rank_local: bool = False,
    model_path: str | None = None,
    trainer_kwargs: dict | None = None,
    retry: bool = False,
):
    """Run ``num_syncs`` weight syncs, checking the received weights after each one, and return
    the per-sync metrics. With ``expect_error`` the first sync must raise it instead; with ``retry``
    generation must then stay blocked, and the syncs run afterwards, starting at the failed step."""
    model_path = os.path.expanduser(model_path or _model)
    ray.init(runtime_env={"env_vars": _UCX_ENV}, ignore_reinit_error=True)
    try:
        ce_config = CheckpointEngineConfig(
            backend="sharded_nixl",
            update_weights_bucket_megabytes=bucket_size_mb,
            engine_kwargs={"sharded_nixl": dict(backend_kwargs)},
        )
        rollout_ce_config = CheckpointEngineConfig(
            backend="sharded_nixl",
            update_weights_bucket_megabytes=bucket_size_mb,
            engine_kwargs={"sharded_nixl": dict(rollout_backend_kwargs or backend_kwargs)},
        )
        model_config = HFModelConfig(path=model_path, use_remove_padding=True)
        # RolloutConfig defaults to TP=2; each mock worker here is one TP=rollout_tp rank
        rollout_config = RolloutConfig(
            name="vllm", tensor_model_parallel_size=rollout_tp, checkpoint_engine=rollout_ce_config
        )

        total = num_trainer + num_rollout
        gpus_per_node = (total + _num_nodes - 1) // _num_nodes
        resource_pool = RayResourcePool(process_on_nodes=[gpus_per_node] * _num_nodes, max_colocate_count=3)
        # every node gets the same bundle count; the remainder is left as an unused split
        spare = gpus_per_node * _num_nodes - total
        sizes = [num_trainer, num_rollout] + ([spare] if spare else [])
        trainer_pool, rollout_pool = split_resource_pool(resource_pool, sizes)[:2]
        actor_wg = create_trainer_worker_group(trainer_pool, model_config, ce_config, **(trainer_kwargs or {}))
        actor_wg.reset()
        rollout, replicas = await create_rollout_worker_group(
            rollout_pool,
            model_config,
            rollout_config,
            check_allclose=True,
            bit_exact=True,
            rollout_layouts=rollout_layouts,
            rank_local=rank_local,
        )
        manager = CheckpointEngineManager(config=ce_config, actor_wg=actor_wg, replicas=replicas)
        if expect_error is not None:
            with pytest.raises(Exception) as ei:
                await manager.update_weights(global_steps=1)
            assert expect_error.__name__ in repr(ei.value) or isinstance(ei.value, expect_error), repr(ei.value)
            if not retry:
                return None
            with pytest.raises(RuntimeError, match="failed weight sync"):
                await manager.resume_generation_replicas()
        all_metrics = []
        for step in range(num_syncs):
            metrics = await manager.update_weights(global_steps=step + 1)
            rollout.check_weights()
            all_metrics.append(metrics)
            print(f"[sharded_nixl] sync {step + 1}: {metrics}")
        return all_metrics
    finally:
        ray.shutdown()


@pytest.mark.skipif(_ngpus < 2, reason="needs >= 2 GPUs")
@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cuda", "cpu"])
# a set: with 4 GPUs the third case is (2, 2) again
@pytest.mark.parametrize("num_trainer, num_rollout", sorted({(1, 1), (2, 2), (2, max(1, _ngpus - 2))}))
@auto_await
async def test_sharded_nixl_correctness(device, num_trainer, num_rollout):
    """Bit-exact weights; metadata is cached after the first sync; every generator receives the
    model once; with several trainers no rank sends the whole payload."""
    if num_trainer + num_rollout > _ngpus * _num_nodes:
        pytest.skip("not enough GPUs")
    metrics = await _run({"device": device, "ownership": "adapter"}, num_trainer, num_rollout)
    first, steady = metrics[0], metrics[-1]
    assert steady["checkpoint_engine/metadata_bytes"] < first["checkpoint_engine/metadata_bytes"]
    model_mb = steady["checkpoint_engine/model_mbytes"]
    assert abs(steady["checkpoint_engine/payload_mbytes"] - model_mb * num_rollout) < 1.0
    if num_trainer > 1:
        assert steady["checkpoint_engine/local_payload_mbytes_max"] < model_mb * num_rollout * 0.75


@pytest.mark.skipif(_ngpus < 2, reason="needs >= 2 GPUs")
@pytest.mark.asyncio
@auto_await
async def test_sharded_nixl_checksum_failure():
    """A byte flipped after hashing must fail the generator's checksum verification."""
    await _run({"_debug_corrupt_slice": True}, 1, 1, expect_error=RuntimeError)


@pytest.mark.skipif(_ngpus < 2, reason="needs >= 2 GPUs")
@pytest.mark.asyncio
@auto_await
async def test_sharded_nixl_retry_after_failure():
    """A sync that fails mid-way keeps generation blocked; retrying the same step repairs every weight."""
    kwargs = {"_debug_corrupt_slice": True, "ack_timeout_s": 60, "notif_timeout_s": 60}
    await _run(kwargs, 1, 1, expect_error=RuntimeError, retry=True, num_syncs=2)


@pytest.mark.skipif(_ngpus < 2, reason="needs >= 2 GPUs")
@pytest.mark.asyncio
@auto_await
async def test_sharded_nixl_version_mismatch():
    """Peers at different sync ids must fail before any weight is applied."""
    await _run(
        {"ownership": "all"},
        1,
        1,
        expect_error=RuntimeError,
        num_syncs=1,
        rollout_backend_kwargs={"ownership": "all", "_debug_sync_offset": 1},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp", "megatron"])
@auto_await
async def test_sharded_nixl_fake_ep_ownership(strategy):
    """Two mock generators own disjoint halves of every layer's experts of a real MoE checkpoint
    (fake EP=2) and must receive exactly those, bit-exact. The Megatron trainer runs TP=2, EP=2
    and sends through the HF-shard export."""
    num_trainer, trainer_kwargs = 1, {}
    if strategy == "megatron":
        pytest.importorskip("megatron.bridge")
        num_trainer = 2
        trainer_kwargs = {"strategy": "megatron", "tensor_model_parallel_size": 2, "expert_model_parallel_size": 2}
    if num_trainer + 2 > _ngpus * _num_nodes:
        pytest.skip("not enough GPUs")
    model_path = os.path.expanduser(_moe_model)
    if not os.path.exists(model_path):
        pytest.skip(f"MoE checkpoint not found: {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    num_layers = int(config.num_hidden_layers)
    num_experts = int(config.num_experts)
    assert num_experts % 2 == 0, num_experts
    experts_per_rank = num_experts // 2
    layouts = [
        {
            "ep_size": 2,
            "ep_rank": rank,
            "local_experts": {
                layer: list(range(rank * experts_per_rank, (rank + 1) * experts_per_rank))
                for layer in range(num_layers)
            },
        }
        for rank in range(2)
    ]
    metrics = await _run(
        {"ownership": "adapter"},
        num_trainer,
        2,
        rollout_layouts=layouts,
        model_path=model_path,
        trainer_kwargs=trainer_kwargs,
    )
    steady = metrics[-1]
    model_mb = steady["checkpoint_engine/model_mbytes"]
    payload_mb = steady["checkpoint_engine/payload_mbytes"]
    assert model_mb < payload_mb < model_mb * 2, (model_mb, payload_mb)


@pytest.mark.skipif(_ngpus < 3, reason="needs >= 3 GPUs")
@pytest.mark.asyncio
@pytest.mark.parametrize("num_trainer", [1, 2])
@auto_await
async def test_sharded_nixl_rank_local_tp2(num_trainer):
    """``rank_local_named_tensors`` with one TP=2 replica of mock workers reporting Qwen3 TP slices
    (dense Qwen3 only): each worker receives exactly its slices, bit-exact, so the payload is the
    rank-local layout once per worker. ``num_trainer=1`` covers trainers < TP (one trainer packs
    both ranks' boxes and grows its staging buffers)."""
    num_rollout = 2
    if num_trainer + num_rollout > _ngpus * _num_nodes:
        pytest.skip("not enough GPUs")
    metrics = await _run(
        {"device": "cuda", "ownership": "adapter"},
        num_trainer,
        num_rollout,
        rollout_tp=2,
        rank_local=True,
    )
    first, steady = metrics[0], metrics[-1]
    assert steady["checkpoint_engine/rank_local"] == 1.0
    assert steady["checkpoint_engine/metadata_bytes"] < first["checkpoint_engine/metadata_bytes"]
    model_mb, layout_mb = steady["checkpoint_engine/model_mbytes"], steady["checkpoint_engine/layout_mbytes"]
    # norms (and any other unsliced tensors) stay full, so slightly more than half the model
    assert model_mb * 0.45 < layout_mb < model_mb * 0.6
    assert abs(steady["checkpoint_engine/payload_mbytes"] - layout_mb * num_rollout) < 1.0
