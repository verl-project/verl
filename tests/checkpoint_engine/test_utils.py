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
import os
from typing import Generator

import ray
import torch
from transformers import AutoModelForCausalLM

from verl.checkpoint_engine import CheckpointEngineRegistry, CheckpointEngineWorker
from verl.checkpoint_engine.sharded_nixl_layout import owned_name_filter, slices_digest
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.workers.config import CheckpointEngineConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.engine.fsdp.utils import unfuse_moe_params
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout import BaseRollout, RolloutReplica
from verl.workers.rollout.base import RolloutLayout

RANK_LOCAL = "rank_local_named_tensors"


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
        if self.checkpoint_engine.send_weights_takes_engine:
            return await self.checkpoint_engine.send_weights(self.engine, global_steps=global_steps)
        per_tensor_param, _ = self.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)


def qwen3_tp_slices(hf_config, tp_size: int, tp_rank: int) -> dict[str, dict]:
    """vLLM's TP slicing of dense Qwen3 (and Qwen2) as a ``RolloutLayout.slices`` table, written
    independently of the engine's planning code.

    Column-parallel (q/k/v and their Qwen2 biases, gate/up, vocab rows) splits dim 0, row-parallel
    (o_proj, down_proj) dim 1; when tp >= num_kv_heads, ``tp // num_kv_heads`` ranks share one kv
    head. Norms stay full.
    """
    cfg = hf_config
    t, r = int(tp_size), int(tp_rank)
    H, nh, nkv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or H // nh
    inter, V, L = cfg.intermediate_size, cfg.vocab_size, cfg.num_hidden_layers
    # Qwen2 configs have no attention_bias: their q/k/v always carry a bias
    qkv_bias = getattr(cfg, "attention_bias", True)
    assert nh % t == 0 and inter % t == 0, (nh, inter, t)

    def rows(shape, n):  # dim-0 block r of size n
        return {"full_shape": list(shape), "box": [[r * n, n], [0, shape[1]]]}

    def cols(shape, n):  # dim-1 block r of size n
        return {"full_shape": list(shape), "box": [[0, shape[0]], [r * n, n]]}

    q, kv = nh * hd, nkv * hd
    if t >= nkv:
        assert t % nkv == 0, (t, nkv)
        blk = r // (t // nkv)
        kv_box = {"full_shape": [kv, H], "box": [[blk * hd, hd], [0, H]]}
    else:
        assert nkv % t == 0, (t, nkv)
        kv_box = rows((kv, H), kv // t)
    out: dict[str, dict] = {}
    if V % t == 0:
        for name in ("model.embed_tokens.weight", "lm_head.weight"):
            out[name] = rows((V, H), V // t)
    for i in range(L):
        p = f"model.layers.{i}."
        out[p + "self_attn.q_proj.weight"] = rows((q, H), q // t)
        out[p + "self_attn.k_proj.weight"] = kv_box
        out[p + "self_attn.v_proj.weight"] = kv_box
        if qkv_bias:
            out[p + "self_attn.q_proj.bias"] = {"full_shape": [q], "box": [[r * (q // t), q // t]]}
            out[p + "self_attn.k_proj.bias"] = {"full_shape": [kv], "box": kv_box["box"][:1]}
            out[p + "self_attn.v_proj.bias"] = out[p + "self_attn.k_proj.bias"]
        out[p + "self_attn.o_proj.weight"] = cols((H, q), q // t)
        out[p + "mlp.gate_proj.weight"] = rows((inter, H), inter // t)
        out[p + "mlp.up_proj.weight"] = rows((inter, H), inter // t)
        out[p + "mlp.down_proj.weight"] = cols((H, inter), inter // t)
    return out


def slice_by_box(t: torch.Tensor, spec: dict) -> torch.Tensor:
    """The local tensor a ``slices`` entry describes: ``t[s0:s0+n0, s1:s1+n1]``."""
    assert list(t.shape) == list(spec["full_shape"]), (t.shape, spec["full_shape"])
    return t[tuple(slice(s, s + n) for s, n in spec["box"])]


def qwen3_tp_layout(hf_config, tp_size: int, tp_rank: int) -> dict:
    """A ``RolloutLayout`` dict advertising ``rank_local_named_tensors`` with Qwen3 TP slices."""
    slices = qwen3_tp_slices(hf_config, tp_size, tp_rank)
    return {
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "wire_formats": ["named_tensors", RANK_LOCAL],
        "slices": slices,
        "slice_digest": slices_digest(slices),
    }


class MockServerAdapter(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        check_allclose: bool = True,
        bit_exact: bool = False,
        rollout_layout: dict | None = None,
    ):
        super().__init__(config, model_config, device_mesh=None)
        self.check_allclose = check_allclose
        self.bit_exact = bit_exact
        self.model = None
        self.received_weights: dict[str, torch.Tensor] = {}
        # what this worker reports from get_weight_ownership, in place of a real vLLM layout
        self.rollout_layout = RolloutLayout.from_dict(rollout_layout)
        self.owns = owned_name_filter(self.rollout_layout)
        self.slices = (self.rollout_layout.slices or {}) if self.rollout_layout is not None else {}

    async def get_weight_ownership(self):
        return self.rollout_layout

    async def resume(self, tags: list[str]):
        raise NotImplementedError()

    async def release(self):
        raise NotImplementedError()

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        async for name, weight in weights:
            weight = weight.clone()
            if self.check_allclose:
                self.received_weights[name] = weight.clone()

    def check_weights(self, wire_format: str = "named_tensors"):
        """Compare the received weights with the HF checkpoint. Weights this worker does not own
        must not arrive; with ``rank_local_named_tensors`` the sliced ones arrive as their box."""
        if not self.check_allclose:
            return
        rank_local = wire_format == RANK_LOCAL
        assert not rank_local or self.slices, "rank-local sync on a worker that reported no slices"

        if self.model is None:
            local_path = copy_to_local(self.model_config.path)
            self.model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16, device_map="cpu")

        model_type = getattr(self.model.config, "model_type", None)
        reference = dict(unfuse_moe_params(self.model.state_dict().items(), model_type))
        checked = 0
        for name, weight in reference.items():
            if not self.owns(name):
                assert name not in self.received_weights, f"weight {name} received but not owned by this worker"
                continue
            assert name in self.received_weights, f"weight {name} not received"
            received = self.received_weights[name]
            if rank_local and name in self.slices:
                weight = slice_by_box(weight, self.slices[name])
            assert received.shape == weight.shape, (
                f"weight {name}: shape {tuple(received.shape)} != {tuple(weight.shape)}"
            )
            expected = weight.to(received.device, dtype=received.dtype)
            if self.bit_exact:
                assert torch.equal(expected, received), f"weight {name} not bit-identical"
            else:
                assert torch.allclose(expected, received), f"weight {name} not equal"
            checked += 1
        self.received_weights.clear()
        print(f"Check passed ({wire_format}), {checked} weights are equal!")


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
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        check_allclose: bool = True,
        bit_exact: bool = False,
        rollout_layouts: list[dict | None] | None = None,
        rank_local: bool = False,
        *args,
        **kwargs,
    ) -> None:
        rank = int(os.environ.get("RANK", "0"))
        layout = rollout_layouts[rank % len(rollout_layouts)] if rollout_layouts else None
        if rank_local:
            # replicas take consecutive workers, so the rank within a TP replica is RANK % tp
            tp = rollout_config.tensor_model_parallel_size
            layout = {**(layout or {}), **qwen3_tp_layout(model_config.hf_config, tp, rank % tp)}
        server_adapter = MockServerAdapter(rollout_config, model_config, check_allclose, bit_exact, layout)
        super().__init__(rollout_config, model_config, server_adapter, *args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_weights(self):
        self.server_adapter.check_weights(getattr(self.checkpoint_engine, "active_wire_format", "named_tensors"))


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
    bit_exact: bool = False,
    rollout_layouts: list[dict | None] | None = None,
    rank_local: bool = False,
) -> tuple[RayWorkerGroup, list[MockReplica]]:
    # create rollout worker group
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(CheckpointEngineWorkerTest),
        model_config=model_config,
        rollout_config=rollout_config,
        check_allclose=check_allclose,
        bit_exact=bit_exact,
        rollout_layouts=rollout_layouts,
        rank_local=rank_local,
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
