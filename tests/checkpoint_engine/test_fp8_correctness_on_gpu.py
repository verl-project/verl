# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import os

import pytest
import ray
import torch

from tests.checkpoint_engine.test_utils import (
    CheckpointEngineWorkerTest,
    MockReplica,
    MockServerAdapter,
    TrainingWorkerTest,
)
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import split_resource_pool
from verl.utils.device import get_device_name
from verl.utils.fp8_utils import FP8QuantizerHelper
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.engine_workers import TrainingWorkerConfig

_ngpus = torch.cuda.device_count()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.0.self_attn.q_proj.weight", True),
        ("model.layers.0.self_attn.k_proj.weight", True),
        ("model.layers.0.self_attn.v_proj.weight", True),
        ("model.layers.0.self_attn.o_proj.weight", True),
        ("model.layers.0.mlp.gate_proj.weight", True),
        ("model.layers.0.mlp.up_proj.weight", True),
        ("model.layers.0.mlp.down_proj.weight", True),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", True),
        ("model.layers.0.mlp.experts.0.up_proj.weight", True),
        ("model.layers.0.mlp.experts.0.down_proj.weight", True),
        ("model.layers.0.mlp.gate.weight", False),
        ("model.embed_tokens.weight", False),
        ("lm_head.weight", False),
        ("model.layers.0.input_layernorm.weight", False),
        ("model.layers.0.post_attention_layernorm.weight", False),
        ("model.norm.weight", False),
        ("model.layers.0.self_attn.q_proj.bias", False),
        ("model.layers.0.fc1.weight", True),
        ("model.layers.0.fc2.weight", True),
    ],
)
def test_should_quantize_param(name, expected):
    quantizer = FP8QuantizerHelper({"weight_block_size": [128, 128]})
    assert quantizer.should_quantize_param(name) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_async_quant_matches_direct():
    from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise

    quantizer = FP8QuantizerHelper({"weight_block_size": [128, 128]})

    torch.manual_seed(42)
    weights = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512, dtype=torch.bfloat16, device="cuda"),
        "model.layers.0.input_layernorm.weight": torch.randn(512, dtype=torch.bfloat16, device="cuda"),
        "model.embed_tokens.weight": torch.randn(1024, 512, dtype=torch.bfloat16, device="cuda"),
    }

    async def run_quant():
        result = {}
        async for name, tensor in quantizer.quant_weights_by_name(weights.items(), dtype=torch.bfloat16):
            result[name] = tensor.clone()
        return result

    result = asyncio.run(run_quant())

    assert result["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    assert "model.layers.0.self_attn.q_proj.weight_scale_inv" in result
    assert result["model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16
    assert result["model.embed_tokens.weight"].dtype == torch.bfloat16

    direct_fp8, direct_scale = scaled_fp8_blockwise(
        weights["model.layers.0.self_attn.q_proj.weight"].to(torch.bfloat16),
        weight_block_size=[128, 128],
    )
    direct_scale = direct_scale.squeeze(-1)
    assert torch.equal(result["model.layers.0.self_attn.q_proj.weight"], direct_fp8)
    assert torch.equal(result["model.layers.0.self_attn.q_proj.weight_scale_inv"], direct_scale)


class TrainingWorkerFP8Test(TrainingWorkerTest):
    def __init__(self, config, checkpoint_engine_config, quant_config):
        super().__init__(config, checkpoint_engine_config)
        self.quant_config = quant_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        per_tensor_param, _ = self.engine.get_per_tensor_param()
        fp8_quantizer = FP8QuantizerHelper(self.quant_config)
        per_tensor_param = fp8_quantizer.quant_weights_by_name(per_tensor_param, dtype=torch.bfloat16)
        await self.checkpoint_engine.send_weights(per_tensor_param)


class FP8MockServerAdapter(MockServerAdapter):
    async def update_weights(self, weights, **kwargs):
        async for name, weight in weights:
            self.received_weights[name] = weight.clone()

    def check_weights(self):
        quantizer = FP8QuantizerHelper({"weight_block_size": [128, 128]})

        fp8_weights = {k for k, v in self.received_weights.items() if v.dtype == torch.float8_e4m3fn}
        scale_weights = {k for k in self.received_weights if k.endswith("_scale_inv")}
        bf16_weights = {
            k
            for k, v in self.received_weights.items()
            if v.dtype != torch.float8_e4m3fn and not k.endswith("_scale_inv")
        }

        for name in fp8_weights:
            assert name + "_scale_inv" in scale_weights, f"Missing scale for {name}"
        for name in fp8_weights:
            assert quantizer.should_quantize_param(name), f"{name} should not be FP8"
        for name in bf16_weights:
            assert not quantizer.should_quantize_param(name), f"{name} should be FP8 but is bf16"

        assert len(fp8_weights) > 0, "No FP8 weights received"
        assert len(fp8_weights) == len(scale_weights), "FP8 weight count != scale count"
        self.received_weights.clear()


class CheckpointEngineWorkerFP8Test(CheckpointEngineWorkerTest):
    def __init__(self, rollout_config, model_config, *args, **kwargs):
        from verl.checkpoint_engine.base import CheckpointEngineWorker

        server_adapter = FP8MockServerAdapter(rollout_config, model_config, check_allclose=False)
        CheckpointEngineWorker.__init__(self, rollout_config, model_config, server_adapter, *args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, _ngpus - 2)])
@auto_await
async def test_fp8_nccl_checkpoint_engine(
    num_trainer,
    num_rollout,
    num_nodes=1,
    num_gpus_per_node=_ngpus,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    model_path = os.path.expanduser(model_path)
    quant_config = {"weight_block_size": [128, 128]}

    ray.init(
        runtime_env={
            "env_vars": {
                "UCX_TLS": "rc,tcp,cuda",
                "UCX_MAX_RNDV_RAILS": "4",
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    checkpoint_engine_config = CheckpointEngineConfig(backend="nccl", engine_kwargs={"nccl": {"rebuild_group": False}})
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(name="vllm", checkpoint_engine=checkpoint_engine_config)

    engine_config = FSDPEngineConfig(forward_only=True, fsdp_size=num_trainer, strategy="fsdp")
    trainer_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
    )
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(TrainingWorkerFP8Test),
        config=trainer_config,
        checkpoint_engine_config=checkpoint_engine_config,
        quant_config=quant_config,
    )
    ray_cls_with_init.update_options(
        {"runtime_env": {"env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}}
    )
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    trainer = RayWorkerGroup(
        resource_pool=trainer_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name()
    )
    trainer.reset()

    rollout_ray_cls = RayClassWithInitArgs(
        cls=ray.remote(CheckpointEngineWorkerFP8Test),
        model_config=model_config,
        rollout_config=rollout_config,
    )
    rollout_wg = RayWorkerGroup(
        resource_pool=rollout_pool, ray_cls_with_init=rollout_ray_cls, device_name=get_device_name()
    )
    rollout_world_size = (
        rollout_config.tensor_model_parallel_size
        * rollout_config.data_parallel_size
        * rollout_config.pipeline_model_parallel_size
    )
    num_replicas = rollout_wg.world_size // rollout_world_size
    replicas = []
    for replica_rank in range(num_replicas):
        replica = MockReplica(replica_rank=replica_rank, config=rollout_config, model_config=model_config)
        replicas.append(replica)
    await asyncio.gather(*[replica.init_hybrid(rollout_wg) for replica in replicas])

    checkpoint_manager = CheckpointEngineManager(config=checkpoint_engine_config, trainer=trainer, replicas=replicas)
    for _ in range(3):
        await checkpoint_manager.update_weights()
        rollout_wg.check_weights()

    ray.shutdown()


if __name__ == "__main__":
    test_fp8_nccl_checkpoint_engine(
        num_trainer=2,
        num_rollout=6,
    )
