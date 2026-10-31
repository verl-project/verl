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
import asyncio
import json
import os

import pytest
import ray
import torch
from transformers import AutoModel

from tests.checkpoint_engine.test_utils import (
    MockReplica,
    TrainingWorkerTest,
    create_rollout_worker_group,
    create_trainer_worker_group,
)
from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineWorker
from verl.checkpoint_engine.delta_sync.hccl_receiver import (
    build_delta_hccl_communicator,
    connect_delta_manifest_sub,
    pump_hccl_delta_session,
)
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import (
    split_resource_pool,
)
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout


@pytest.mark.asyncio
@pytest.mark.parametrize("rebuild_group", [False])
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 6)])
# 128MB bucket is smaller than the largest weight of Qwen3-8B (embed_tokens, ~1.2GB in bf16),
# so it exercises the chunked transfer path for weights larger than the bucket.
@pytest.mark.parametrize("bucket_size_mb", [3072, 128])
@auto_await
async def test_hccl_checkpoint_engine(
    rebuild_group,
    num_trainer,
    num_rollout,
    bucket_size_mb,
    num_nodes=1,
    num_gpus_per_node=8,
    check_allclose=True,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    model_path = os.path.expanduser(model_path)
    ray.init(
        runtime_env={
            "env_vars": {
                "HCCL_CONNECT_TIMEOUT": "1500",
                "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
                "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    # initialize config
    checkpoint_engine_config = CheckpointEngineConfig(
        backend="nccl",
        engine_kwargs={"nccl": {"rebuild_group": rebuild_group}},
        update_weights_bucket_megabytes=bucket_size_mb,
    )
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(name="vllm", checkpoint_engine=checkpoint_engine_config)

    # create trainer and rollout worker group
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
    resource_pool.get_placement_groups(device_name=get_device_name())
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    actor_wg = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    actor_wg.reset()
    rollout, replicas = await create_rollout_worker_group(rollout_pool, model_config, rollout_config, check_allclose)

    # create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(config=checkpoint_engine_config, actor_wg=actor_wg, replicas=replicas)
    for _ in range(3):
        await checkpoint_manager.update_weights()
        rollout.check_weights()

    ray.shutdown()


@pytest.mark.skip(reason="temporary skip since our ci environment is not ready")
@pytest.mark.asyncio
@pytest.mark.parametrize("rebuild_group", [False])
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 6)])
@auto_await
async def test_kimi_checkpoint_engine(
    rebuild_group,
    num_trainer,
    num_rollout,
    num_nodes=1,
    num_gpus_per_node=8,
    check_allclose=True,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    model_path = os.path.expanduser(model_path)
    ray.init(
        runtime_env={
            "env_vars": {
                "HCCL_CONNECT_TIMEOUT": "1500",
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    # initialize config
    checkpoint_engine_config = CheckpointEngineConfig(
        backend="kimi_ckpt_engine", engine_kwargs={"kimi_ckpt_engine": {"rebuild_group": rebuild_group}}
    )
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(name="vllm", checkpoint_engine=checkpoint_engine_config)

    # create trainer and rollout worker group
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
    resource_pool.get_placement_groups(device_name=get_device_name())
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    actor_wg = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    actor_wg.reset()
    rollout, replicas = await create_rollout_worker_group(rollout_pool, model_config, rollout_config, check_allclose)

    # create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(config=checkpoint_engine_config, actor_wg=actor_wg, replicas=replicas)
    for _ in range(3):
        await checkpoint_manager.update_weights()
        rollout.check_weights()

    ray.shutdown()


@pytest.mark.skip(reason="temporary skip since our ci environment is not ready")
@pytest.mark.asyncio
@pytest.mark.parametrize("rebuild_group", [False])
@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 6)])
@auto_await
async def test_mooncake_checkpoint_engine(
    rebuild_group,
    num_trainer,
    num_rollout,
    device,
    num_nodes=1,
    num_gpus_per_node=8,
    check_allclose=True,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    model_path = os.path.expanduser(model_path)
    ray.init(
        runtime_env={
            "env_vars": {
                "ASCEND_USE_SHORT_CONNECTION": "1",
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    # initialize config
    checkpoint_engine_config = CheckpointEngineConfig(
        backend="mooncake", engine_kwargs={"mooncake": {"device": device, "rebuild_group": rebuild_group}}
    )
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(name="vllm", checkpoint_engine=checkpoint_engine_config)

    # create trainer and rollout worker group
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
    resource_pool.get_placement_groups(device_name=get_device_name())
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    actor_wg = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    actor_wg.reset()
    rollout, replicas = await create_rollout_worker_group(rollout_pool, model_config, rollout_config, check_allclose)

    # create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(config=checkpoint_engine_config, actor_wg=actor_wg, replicas=replicas)
    for _ in range(3):
        await checkpoint_manager.update_weights()
        rollout.check_weights()

    ray.shutdown()


class TrainingWorkerDeltaTest(TrainingWorkerTest):
    """Trainer-side test worker with the engine-driven delta dispatch."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        if mode == "delta_sharded":
            # parity with engine_workers.py: the delta engine owns the seed/
            # steady state machine and drives the training engine itself.
            metrics = await self.checkpoint_engine.send_weights(self.engine, global_steps=global_steps)
            return metrics or {}
        return await super().update_weights(global_steps=global_steps, mode=mode)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def perturb_positive_half(self):
        """Deterministic sparse perturbation: every positive element := 2.0.

        The predicate is element-wise (value-based) and the assigned constant
        is exactly representable in bf16, so applying it independently on every
        rank's FSDP shard equals applying it once on the full model -- the
        rollout-side reference computes the identical result from the base
        weights, and the positive set is invariant across perturbation rounds.
        """
        for _, param in self.engine.module.named_parameters():
            if not param.is_floating_point():
                continue
            local = param.to_local() if hasattr(param, "to_local") else param
            with torch.no_grad():
                local[local > 0] = 2.0


class DeltaHCCLMockServerAdapter(BaseRollout):
    """Direct-receiver mock standing in for the vLLM worker engines: joins the trainer's PyHccl
    broadcast group under the rank the single-sender topology assigned to this
    CheckpointEngineWorker, pulls each session's flushes and applies them to a
    local weight dict with the wire contract's semantics (checksum verify,
    dense overwrite / sparse index_copy_). The expected state is the disk model
    with the same deterministic perturbation applied, compared bit-exactly.
    """

    def __init__(self, rollout_config, model_config, check_allclose: bool = True):
        super().__init__(rollout_config, model_config, device_mesh=None)
        self.check_allclose = check_allclose
        self.applied: dict[str, torch.Tensor] = {}
        self.perturb_rounds = 0
        self._sub = None
        self._comm = None

    def _device(self):
        return torch.device("npu", torch.npu.current_device())

    async def resume(self, tags: list[str]):
        raise NotImplementedError()

    async def release(self):
        raise NotImplementedError()

    async def update_weights(self, weights, global_steps=None, wire_format="delta_flush", **kwargs):
        from verl.workers.rollout.utils import ensure_async_iterator

        assert wire_format == "delta_flush"
        flushes = ensure_async_iterator(weights)
        first_named, _ = await anext(flushes)
        info = _decode_rendezvous(first_named)
        assert info is not None, "expected the HCCL rendezvous control item"
        if self._comm is None:
            # SUB before communicator, and the trainer publishes the first
            # manifest only after every receiver's wire-ready flag is up, so
            # subscribing before joining cannot miss any.
            self._sub = connect_delta_manifest_sub(info["zmq_ip"], info["zmq_port"], info["topic"])
            self._comm = build_delta_hccl_communicator(
                info["dist_ip"], info["dist_port"], info["engine_rank"], info["world_size"], self._device()
            )
        pump_hccl_delta_session(self._sub, self._comm, self._device(), on_flush=self._apply_flush)
        async for _ in flushes:
            pass

    def _apply_flush(self, named: list[tuple[str, torch.Tensor]]) -> None:
        from verl.checkpoint_engine.delta_sync.decode import decode_delta_wire

        encoding, patches = decode_delta_wire(named)
        for patch in patches:
            if encoding == "dense":
                # clone: the broadcast buffers are recycled after this callback
                self.applied[patch.name] = patch.values.detach().clone().view(patch.shape)
            else:
                flat = self.applied[patch.name].view(-1)
                flat.index_copy_(0, patch.positions.to(torch.int64), patch.values.to(flat.dtype))

    def perturb_reference(self):
        self.perturb_rounds += 1

    def check_weights(self):
        if not self.check_allclose:
            return
        if not self.applied:
            raise AssertionError("no delta flushes were applied")
        local_path = copy_to_local(self.model_config.path)
        # The trainer sends full-model HF names carrying the "model." wrapper
        # (e.g. "model.language_model.embed_tokens.weight" on a VLM); AutoModel
        # returns the backbone state dict without that wrapper, so strip it from
        # the applied names to align the two. AutoModel also covers VLM
        # checkpoints (vision tower included), unlike AutoModelForCausalLM.
        reference = AutoModel.from_pretrained(local_path, torch_dtype=torch.bfloat16, device_map="cpu")
        expected = reference.state_dict()
        applied_stripped = {k.removeprefix("model."): v for k, v in self.applied.items()}
        perturbed = self.perturb_rounds > 0
        for name, base in expected.items():
            assert name in applied_stripped, f"weight {name} not received (applied has {len(applied_stripped)} weights)"
            got = applied_stripped[name]
            if not perturbed:
                assert torch.equal(got.cpu(), base), f"weight {name} differs from the base model"
                continue
            base_pos = base > 0
            ok = torch.where(base_pos, got.cpu() == 2.0, got.cpu() == base)
            assert bool(ok.all()), f"weight {name} differs from the perturbed expectation"
        del reference
        print("Delta check passed, all weights are bit-exact!")


class DeltaHCCLRolloutWorkerTest(CheckpointEngineWorker):
    def __init__(self, rollout_config, model_config, check_allclose=True, *args, **kwargs):
        server_adapter = DeltaHCCLMockServerAdapter(rollout_config, model_config, check_allclose)
        super().__init__(rollout_config, model_config, server_adapter, *args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_weights(self):
        self.server_adapter.check_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def perturb_reference(self):
        self.server_adapter.perturb_reference()


def _decode_rendezvous(named_tensors) -> dict | None:
    for name, tensor in named_tensors:
        if name == "__delta_rendezvous__":
            return json.loads(bytes(tensor.detach().cpu().numpy().tobytes()).decode())
    return None


async def create_delta_rollout_worker_group(resource_pool, model_config, rollout_config, check_allclose=True):
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(DeltaHCCLRolloutWorkerTest),
        model_config=model_config,
        rollout_config=rollout_config,
        check_allclose=check_allclose,
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name())
    rollout_world_size = (
        rollout_config.tensor_model_parallel_size
        * rollout_config.data_parallel_size
        * rollout_config.pipeline_model_parallel_size
    )
    num_replicas = wg.world_size // rollout_world_size
    replicas = [
        MockReplica(replica_rank=replica_rank, config=rollout_config, model_config=model_config)
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[replica.init_hybrid(wg) for replica in replicas])
    return wg, replicas


@pytest.mark.asyncio
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 6)])
# 128MB bucket is smaller than the largest weight of Qwen3-8B, so it exercises
# the small-bucket chunked transfer path on the seed and the steady flushes.
@pytest.mark.parametrize("bucket_size_mb", [3072, 128])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@auto_await
async def test_delta_hccl_checkpoint_engine(
    num_trainer,
    num_rollout,
    bucket_size_mb,
    tensor_parallel_size,
    num_nodes=1,
    num_gpus_per_node=8,
    check_allclose=True,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    model_path = os.path.expanduser(model_path)
    ray.init(
        runtime_env={
            "env_vars": {
                "HCCL_CONNECT_TIMEOUT": "1500",
                "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
                "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
                "VERL_LOGGING_LEVEL": "DEBUG",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            }
        }
    )

    try:
        checkpoint_engine_config = CheckpointEngineConfig(
            backend="delta_sharded",
            engine_kwargs={"delta_sharded": {}},
            update_weights_bucket_megabytes=bucket_size_mb,
        )
        model_config = HFModelConfig(path=model_path, use_remove_padding=True)
        rollout_config = RolloutConfig(
            name="vllm",
            tensor_model_parallel_size=tensor_parallel_size,
            checkpoint_engine=checkpoint_engine_config,
        )

        resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
        resource_pool.get_placement_groups(device_name=get_device_name())
        trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
        actor_wg = create_trainer_worker_group(
            trainer_pool, model_config, checkpoint_engine_config, worker_cls=TrainingWorkerDeltaTest
        )
        actor_wg.reset()
        rollout, replicas = await create_delta_rollout_worker_group(
            rollout_pool, model_config, rollout_config, check_allclose
        )

        checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=actor_wg, replicas=replicas
        )

        # 1. seed sync: dense values-only wire, full-coverage metrics.
        metrics = await checkpoint_manager.update_weights()
        assert metrics.get("checkpoint_engine/changed_ratio") == 1.0, metrics
        assert metrics.get("checkpoint_engine/flushes", 0) >= 1, metrics
        rollout.check_weights()

        # 2. steady sync after a deterministic sparse perturbation.
        actor_wg.perturb_positive_half()
        rollout.perturb_reference()
        metrics = await checkpoint_manager.update_weights()
        ratio = metrics.get("checkpoint_engine/changed_ratio")
        assert ratio is not None, "perturbed steady sync must report changed_ratio"
        # the perturbation touches every positive element (~half of a random init)
        assert 0.3 < ratio < 0.7, f"changed_ratio {ratio} outside the constructed sparsity range"
        assert metrics.get("checkpoint_engine/changed_elems", 0) > 0, metrics
        assert metrics.get("checkpoint_engine/payload_mbytes", 0) > 0, metrics
        assert metrics.get("checkpoint_engine/flushes", 0) >= 1, metrics
        rollout.check_weights()

        # 3. steady sync with zero changes: the session is terminal-only (zero
        #    flushes), but entries still flow with zero counts to keep the gather
        #    in lockstep, so the metrics report zero values rather than absent keys.
        metrics = await checkpoint_manager.update_weights()
        assert metrics.get("checkpoint_engine/changed_ratio") == 0.0, metrics
        assert metrics.get("checkpoint_engine/changed_elems") == 0.0, metrics
        assert metrics.get("checkpoint_engine/flushes") == 0.0, metrics
        rollout.check_weights()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    test_hccl_checkpoint_engine(
        rebuild_group=False,
        num_trainer=2,
        num_rollout=6,
        num_nodes=1,
        num_gpus_per_node=8,
        check_allclose=False,
        model_path=os.environ["HDFS_ROOT"] + "/model/Qwen3-30B-A3B-Base",
    )
    test_delta_hccl_checkpoint_engine(
        num_trainer=2,
        num_rollout=6,
        bucket_size_mb=3072,
        tensor_parallel_size=1,
        num_nodes=1,
        num_gpus_per_node=8,
        check_allclose=False,
        model_path=os.environ["HDFS_ROOT"] + "/model/Qwen3-30B-A3B-Base",
    )
