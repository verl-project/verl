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
import hashlib
import json
import os
from pathlib import Path
from typing import Generator

import ray
import torch
from transformers import AutoModelForCausalLM

from verl.checkpoint_engine import CheckpointEngineRegistry, CheckpointEngineWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.workers.config import CheckpointEngineConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout import BaseRollout, RolloutReplica

_DIAGNOSTIC_WEIGHTS = {"model.layers.0.mlp.up_proj.weight", "model.layers.1.mlp.down_proj.weight"}


def _weight_diagnostics(weight: torch.Tensor) -> dict:
    """Summarize on CPU so diagnostics do not depend on NPU comparison kernels."""
    cpu = weight.detach().cpu().contiguous()
    values = cpu.float().flatten()
    return {
        "shape": tuple(weight.shape),
        "dtype": str(weight.dtype),
        "device": str(weight.device),
        "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "bf16_sha256": hashlib.sha256(cpu.bfloat16().view(torch.uint8).numpy().tobytes()).hexdigest(),
        "nan_count": torch.isnan(values).sum().item(),
        "inf_count": torch.isinf(values).sum().item(),
        "first_values": values[:8].tolist(),
    }


def _checkpoint_diagnostics(local_path: str, name: str, received: torch.Tensor, context: str):
    """Read only the original tensor, avoiding a second full FP32 reference model."""
    from safetensors import safe_open

    try:
        root = Path(local_path)
        index = root / "model.safetensors.index.json"
        if index.is_file():
            with index.open(encoding="utf-8") as handle:
                shard = json.load(handle)["weight_map"][name]
        else:
            shard = "model.safetensors"
        with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
            original = handle.get_tensor(name)
        actual = received.detach().cpu()
        reference = original.to(actual.dtype)
        rounded = original.bfloat16().to(actual.dtype)
        print(f"{context} checkpoint_original: {_weight_diagnostics(original)}", flush=True)
        print(f"{context} checkpoint_as_received_dtype: {_weight_diagnostics(reference)}", flush=True)
        print(
            f"{context} checkpoint_comparison: "
            f"original_equal={torch.equal(reference, actual)} "
            f"original_allclose={torch.allclose(reference, actual)} "
            f"bf16_roundtrip_equal={torch.equal(rounded, actual)} "
            f"bf16_roundtrip_allclose={torch.allclose(rounded, actual)} "
            f"original_changed_by_bf16={(reference != rounded).sum().item()}/{reference.numel()}",
            flush=True,
        )
    except Exception as exc:
        # Diagnostics must not obscure the original assertion (e.g. a .bin checkpoint).
        print(f"{context} checkpoint_diagnostics_unavailable={type(exc).__name__}: {exc}", flush=True)


class TrainingWorkerTest(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig, checkpoint_engine_config: CheckpointEngineConfig) -> None:
        super().__init__(config)

        backend = checkpoint_engine_config.backend
        bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
        engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
        if torch.distributed.get_rank() == 0:
            engine_kwargs["is_master"] = True
        self.checkpoint_engine = CheckpointEngineRegistry.new(backend, bucket_size=bucket_size, **engine_kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        per_tensor_param, _ = self.engine.get_per_tensor_param()
        sender_diagnostics = os.environ.get("VERL_CHECKPOINT_SENDER_DIAGNOSTICS", "off")
        if sender_diagnostics not in {"off", "hash"}:
            raise ValueError(f"Invalid VERL_CHECKPOINT_SENDER_DIAGNOSTICS: {sender_diagnostics!r}")
        if sender_diagnostics == "off":
            # Preserve the original generator path: no tensor reads, copies, or extra synchronization.
            await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
            return

        def diagnostic_params():
            for name, weight in per_tensor_param:
                if name in _DIAGNOSTIC_WEIGHTS and torch.distributed.get_rank() == 0:
                    # This CPU copy synchronizes the target tensor and can affect race timing.
                    print(
                        f"Checkpoint diagnostic sender pid={os.getpid()} rank=0 step={global_steps} "
                        f"bucket_bytes={getattr(self.checkpoint_engine, 'bucket_size', None)} name={name}: "
                        f"{_weight_diagnostics(weight)}",
                        flush=True,
                    )
                yield name, weight

        await self.checkpoint_engine.send_weights(diagnostic_params(), global_steps=global_steps)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)


class MockServerAdapter(BaseRollout):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, check_allclose: bool = True):
        super().__init__(config, model_config, device_mesh=None)
        self.check_allclose = check_allclose
        self.model = None
        self.global_steps = None
        self.received_weights: dict[str, torch.Tensor] = {}

    async def resume(self, tags: list[str]):
        raise NotImplementedError()

    async def release(self):
        raise NotImplementedError()

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        self.global_steps = kwargs.get("global_steps")
        async for name, weight in weights:
            weight = weight.clone()
            if self.check_allclose:
                self.received_weights[name] = weight.clone()

    def check_weights(self):
        if not self.check_allclose:
            return

        if self.model is None:
            local_path = copy_to_local(self.model_config.path)
            self.model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16, device_map="cpu")

        for name, weight in self.model.state_dict().items():
            assert name in self.received_weights, f"weight {name} not received"
            received = self.received_weights[name]
            expected = weight.to(received.device, dtype=received.dtype)
            matches = torch.allclose(expected, received)
            # Keep both experiment modes identical on the receiver. Diagnose any failing
            # tensor, not just the sampled sender names, only after observing the failure.
            if not matches:
                context = (
                    f"Checkpoint diagnostic receiver pid={os.getpid()} "
                    f"rank={torch.distributed.get_rank()} step={self.global_steps} "
                    f"sender_diagnostics={os.environ.get('VERL_CHECKPOINT_SENDER_DIAGNOSTICS', 'off')} "
                    f"bucket_mb={self.config.checkpoint_engine.update_weights_bucket_megabytes} "
                    f"model={self.model_config.path} name={name}"
                )
                print(f"{context} reference: {_weight_diagnostics(weight)}", flush=True)
                print(f"{context} received: {_weight_diagnostics(received)}", flush=True)
                _checkpoint_diagnostics(copy_to_local(self.model_config.path), name, received, context)
                # Build the CPU reference independently of the device-side conversion.
                expected_cpu = weight.detach().cpu().to(received.dtype).float().flatten()
                received_cpu = received.detach().cpu().float().flatten()
                mismatches = ~torch.isclose(expected_cpu, received_cpu)
                finite = torch.isfinite(expected_cpu) & torch.isfinite(received_cpu)
                abs_diff = (expected_cpu - received_cpu).abs()
                first = int(mismatches.to(torch.uint8).argmax().item())
                print(
                    f"{context} comparison: device_allclose={matches} "
                    f"cpu_allclose={torch.allclose(expected_cpu, received_cpu)} "
                    f"mismatch_count={mismatches.sum().item()}/{received_cpu.numel()} "
                    f"finite_pair_count={finite.sum().item()} "
                    f"max_finite_abs_diff={torch.where(finite, abs_diff, 0.0).max().item()} "
                    f"bf16_allclose={torch.allclose(expected_cpu.bfloat16(), received_cpu.bfloat16())} "
                    f"first_mismatch_flat_index={first if mismatches.any() else None} "
                    f"expected_window={expected_cpu[first : first + 8].tolist()} "
                    f"received_window={received_cpu[first : first + 8].tolist()}",
                    flush=True,
                )
                raise AssertionError(f"weight {name} not equal")
        self.received_weights.clear()
        print("Check passed, all weights are equal!")


class MockReplica(RolloutReplica):
    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        raise NotImplementedError

    async def launch_servers(self):
        """Launch http server in each node."""
        raise NotImplementedError


class CheckpointEngineWorkerTest(CheckpointEngineWorker):
    def __init__(
        self, rollout_config: RolloutConfig, model_config: HFModelConfig, check_allclose: bool = True, *args, **kwargs
    ) -> None:
        server_adapter = MockServerAdapter(rollout_config, model_config, check_allclose)
        super().__init__(rollout_config, model_config, server_adapter, *args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_weights(self):
        self.server_adapter.check_weights()


def create_trainer_worker_group(
    resource_pool: RayResourcePool, model_config: HFModelConfig, checkpoint_engine_config: CheckpointEngineConfig
) -> RayWorkerGroup:
    engine_config = FSDPEngineConfig(forward_only=True, fsdp_size=resource_pool.world_size, strategy="fsdp")
    trainer_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
    )

    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(TrainingWorkerTest),
        config=trainer_config,
        checkpoint_engine_config=checkpoint_engine_config,
    )
    ray_cls_with_init.update_options(
        {
            "runtime_env": {
                "env_vars": {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            }
        }
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name())
    return wg


async def create_rollout_worker_group(
    resource_pool: RayResourcePool,
    model_config: HFModelConfig,
    rollout_config: RolloutConfig,
    check_allclose: bool = True,
) -> tuple[RayWorkerGroup, list[MockReplica]]:
    # create rollout worker group
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(CheckpointEngineWorkerTest),
        model_config=model_config,
        rollout_config=rollout_config,
        check_allclose=check_allclose,
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name())

    # create rollout replicas
    rollout_world_size = (
        rollout_config.tensor_model_parallel_size
        * rollout_config.data_parallel_size
        * rollout_config.pipeline_model_parallel_size
    )
    num_replicas = wg.world_size // rollout_world_size
    replicas = []
    for replica_rank in range(num_replicas):
        replica = MockReplica(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
        )
        replicas.append(replica)
    await asyncio.gather(*[replica.init_hybrid(wg) for replica in replicas])

    return wg, replicas
