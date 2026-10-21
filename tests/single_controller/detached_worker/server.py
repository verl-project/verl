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
"""
Server starts a Trainer. Client sends data to the server to train.
"""

import os

os.environ["MEGATRON_USE_CUDA_TIMER"] = "0"
os.environ["MEGATRON_START_PROCESS_TIMER"] = "False"
os.environ["NCCL_DEBUG"] = "WARN"

import ray
import torch
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads
from tensordict import TensorDict
from transformers import LlamaConfig

from verl import DataProto
from verl.models.mcore.bridge import AutoBridge
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config
from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
from verl.workers.config.optimizer import McoreOptimizerConfig


@ray.remote
class Trainer(Worker):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        hf_config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            vocab_size=256,
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
        )
        # Build a randomly initialized model without downloading HF weights.
        bridge = AutoBridge.from_hf_config(hf_config)
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.tensor_model_parallel_size = 2
        provider.pipeline_model_parallel_size = 1
        provider.virtual_pipeline_model_parallel_size = None
        provider.context_parallel_size = 1
        provider.expert_model_parallel_size = 1
        provider.expert_tensor_parallel_size = 1
        provider.sequence_parallel = True
        provider.params_dtype = torch.bfloat16
        provider.pipeline_dtype = torch.bfloat16
        provider.bf16 = True
        provider.fp16 = False
        provider.finalize()
        provider.initialize_model_parallel(seed=10)

        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=mpu.get_data_parallel_rank(),
            is_collect=mpu.get_tensor_model_parallel_rank() == 0,
        )
        self.module, _ = make_megatron_module(
            wrap_config=McoreModuleWrapperConfig(wrap_with_ddp=True, use_distributed_optimizer=True),
            hf_config=hf_config,
            bridge=bridge,
            provider=provider,
            override_ddp_config={"overlap_grad_reduce": False, "overlap_param_gather": False},
        )
        self.model = self.module[0]
        self.model.train()
        optim_config = init_megatron_optim_config(
            McoreOptimizerConfig(lr=1e-6, clip_grad=1.0), use_distributed_optimizer=True, bf16=True
        )
        self.optimizer = get_megatron_optimizer(model=self.module, config=optim_config)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"))
    def train_model(self, data: DataProto) -> DataProto:
        data = data.to(torch.device("cuda", torch.cuda.current_device()))
        input_ids = data.batch["input_ids"]
        if not data.batch["attention_mask"].bool().all():
            raise ValueError("This example expects unpadded sequences.")

        self.optimizer.zero_grad()
        self.model.zero_grad_buffer()
        # MCore returns per-token losses when labels are supplied. Its attention
        # layer applies the causal mask; this example has no padding.
        token_losses = self.model(
            input_ids=input_ids,
            position_ids=data.batch["position_ids"],
            attention_mask=None,
            labels=torch.roll(input_ids, shifts=-1, dims=-1),
        )
        # Exclude the final token, whose rolled label belongs to the sequence start.
        loss = token_losses[:, :-1].mean()
        self.optimizer.scale_loss(loss).backward()
        finalize_model_grads(self.module)
        update_successful, _, _ = self.optimizer.step()
        if not update_successful:
            raise RuntimeError("Megatron optimizer step failed.")

        losses = token_losses[:, :-1].detach().mean(dim=-1).cpu()
        return DataProto(batch=TensorDict({"loss": losses}, batch_size=[input_ids.shape[0]]))


if __name__ == "__main__":
    ray.init(address="auto", namespace="verl")

    resource_pool = RayResourcePool(process_on_nodes=[2], detached=True)
    cls_with_init_args = RayClassWithInitArgs(cls=Trainer)
    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=cls_with_init_args,
        name_prefix="trainer",
        detached=True,
    )

    worker_group.init_model()
    print(worker_group.worker_names)
