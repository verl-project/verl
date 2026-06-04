# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import os

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as McoreDDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from verl.utils.megatron_utils import load_megatron_optimizer, offload_megatron_optimizer

# ==== Helper functions ==== #


MICROBATCH_SIZE = 32
SEQUENCE_LENGTH = 64


@pytest.fixture
def initialize_distributed_env():
    """Bring up and tear down torch.distributed + Megatron parallel state
    before / after each test."""

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    torch.cuda.set_device(0)
    assert not dist.is_initialized()
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    mpu.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)

    try:
        yield
    finally:
        dist.barrier()
        dist.destroy_process_group()
        mpu.destroy_model_parallel()


def init_model():
    """Initialize a small GPT model for the test, wrapped in Megatron DDP."""

    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=SEQUENCE_LENGTH,
    ).cuda()

    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model_chunk = McoreDDP(transformer_config, ddp_config, gpt_model)
    return [model_chunk]


def init_optimizer(model, use_precision_aware_optimizer):
    """Initialize an optimizer for the model."""

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-6,
        min_lr=1e-6,
        clip_grad=1.0,
        weight_decay=0.0,
        use_distributed_optimizer=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_precision_aware_optimizer=use_precision_aware_optimizer,
    )
    return get_megatron_optimizer(optimizer_config, model)


def optimizer_state_is_on_device(
    optimizer,
    device,
    use_precision_aware_optimizer,
):
    """Check that all tensors inside optimizer_state are on the specified device."""

    opts = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]

    # If use_precision_aware_optimizer=True, verify that "master_param" is
    # populated for each parameter and not shard_fp32_from_float16_groups
    # (this is an assumption made by VeRL's optimizer offloading code).
    if use_precision_aware_optimizer:
        for opt in opts:
            for group in opt.shard_fp32_from_float16_groups:
                for param in group:
                    assert param is None
            param_to_param_opt_state = opt.optimizer.state
            for param_state in param_to_param_opt_state.values():
                assert param_state.get("master_param", None) is not None

    # Check device placement of optimizer state.
    for opt in opts:
        # Check if master params are on the requested device when
        # use_precision_aware_optimizer=True.
        if not use_precision_aware_optimizer:
            for group in opt.shard_fp32_from_float16_groups:
                for param in group:
                    if isinstance(param, torch.Tensor) and param.device != device:
                        return False
        # Check whether any parameters are not on the expected device.
        param_to_param_opt_state = opt.optimizer.state
        for param_state in param_to_param_opt_state.values():
            for v in param_state.values():
                if isinstance(v, torch.Tensor) and v.device != device:
                    return False

    return True


# ==== Tests ==== #


@pytest.mark.parametrize("use_precision_aware_optimizer", [False, True])
def test_distributed_optimizer_offload_and_load(
    initialize_distributed_env,
    use_precision_aware_optimizer,
):
    # Initialize model and optimizer.
    model_chunks = init_model()
    optimizer = init_optimizer(model_chunks, use_precision_aware_optimizer)

    # Fully initialize the optimizer state by calling optimizer.step() on
    # dummy gradients set to 0.
    for model_chunk in model_chunks:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad(set_to_none=False)
    update_successful, _, _ = optimizer.step()
    assert update_successful

    # Offload optimizer state.
    offload_megatron_optimizer(optimizer)

    # Make sure everything has been offloaded.
    assert optimizer_state_is_on_device(
        optimizer,
        torch.device("cpu"),
        use_precision_aware_optimizer,
    )

    # Load optimizer state.
    load_megatron_optimizer(optimizer)

    # Make sure everything has been loaded.
    assert optimizer_state_is_on_device(
        optimizer,
        torch.device("cuda:0"),
        use_precision_aware_optimizer,
    )
