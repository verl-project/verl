# Copyright 2026 Amazon.com Inc and/or its affiliates
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
import torch.distributed
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, GptOssConfig, Qwen2Config

from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.fsdp_utils import (
    MixedPrecisionPolicy,
    apply_fsdp2,
    get_fsdp_wrap_policy,
    merged_lora_context,
)


def _test_merged_lora_context_worker(
    rank, world_size, rendezvous_file, strategy, model_config, lora_config_dict, backup_adapters
):
    """Worker function for testing merged_lora_context with FSDP.

    Args:
        rank: Process rank
        world_size: Total number of processes
        rendezvous_file: Path to rendezvous file for distributed init
        strategy: FSDP strategy ("fsdp" or "fsdp2")
        model_config: Model configuration object (Qwen2Config, GptOssConfig, etc.)
        lora_config_dict: Dictionary of LoRA configuration parameters
        backup_adapters: Whether to backup adapter weights before merging
    """
    get_torch_device().set_device(rank)
    torch.distributed.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    device_mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

    # Create model from provided config
    with torch.device(get_device_name()):
        model = AutoModelForCausalLM.from_config(
            config=model_config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model = model.to(device=get_device_name())

    # Add LoRA with provided config
    lora_config = LoraConfig(**lora_config_dict)
    model = get_peft_model(model, lora_config)

    # Initialize LoRA adapter weights to non-zero values for testing
    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                for adapter_name in module.lora_A.keys():
                    if adapter_name in module.lora_A:
                        # Initialize lora_A with values around 1.0
                        module.lora_A[adapter_name].weight.data.uniform_(0.5, 1.5)
                    if adapter_name in module.lora_B:
                        # Initialize lora_B with values around 2.0
                        module.lora_B[adapter_name].weight.data.uniform_(1.5, 2.5)

    # Wrap model with FSDP
    if strategy == "fsdp":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )
        model = FSDP(
            model,
            use_orig_params=True,
            device_id=get_torch_device().current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
            auto_wrap_policy=get_fsdp_wrap_policy(module=model, is_lora=True),
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        )
        fsdp_kwargs = {
            "mesh": device_mesh,
            "mp_policy": mp_policy,
        }
        apply_fsdp2(model, fsdp_kwargs, {})

    # Test: backup adapter weights, merge, restore
    from peft.tuners.lora import LoraLayer

    lora_layers = [m for m in model.modules() if isinstance(m, LoraLayer)]

    # Verify LoRA layers exist
    assert len(lora_layers) > 0, "Model should have LoRA layers"

    # Initially not merged
    for layer in lora_layers:
        assert not getattr(layer, "merged", False), "LoRA should not be merged initially"

    # Backup adapter weights before merge
    from peft.utils.save_and_load import get_peft_model_state_dict

    original_adapter_weights = get_peft_model_state_dict(model)

    # Use merged_lora_context with the specified backup_adapters flag
    for _ in range(3):
        with merged_lora_context(model, backup_adapters=backup_adapters):
            # Inside context, LoRA should be merged
            for layer in lora_layers:
                assert getattr(layer, "merged", False), "LoRA should be merged inside context"

    # After context, check the state based on backup_adapters flag
    for layer in lora_layers:
        assert not getattr(layer, "merged", False), "LoRA should be unmerged after context"

    restored_adapter_weights = get_peft_model_state_dict(model)

    # Verify adapter weights are restored exactly
    for key in original_adapter_weights.keys():
        assert key in restored_adapter_weights, f"Key {key} should be in restored weights"
        torch.testing.assert_close(
            original_adapter_weights[key].cpu(),
            restored_adapter_weights[key].cpu(),
            rtol=1e-5,
            atol=1e-6,
            msg=f"Adapter weight {key} should be restored to original value",
        )

    if rank == 0:
        model_name = model_config.__class__.__name__
        backup_mode = "with backup" if backup_adapters else "without backup"
        print(f"merged_lora_context test with {model_name} {strategy} {backup_mode} passed on {world_size} GPUs!")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2,))
@pytest.mark.parametrize("strategy", ("fsdp", "fsdp2"))
@pytest.mark.parametrize("backup_adapters", (True, False))
def test_merged_lora_context_qwen2(world_size, strategy, backup_adapters, tmp_path):
    """Test merged_lora_context with FSDP on Qwen2 model."""
    rendezvous_file = str(tmp_path / f"rdzv_file_qwen2_{backup_adapters}")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    # Create Qwen2 model config
    model_config = Qwen2Config(num_hidden_layers=2, num_attention_heads=2, hidden_size=128)

    # Create LoRA config for Qwen2
    lora_config_dict = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    mp.spawn(
        fn=_test_merged_lora_context_worker,
        args=(world_size, rendezvous_file, strategy, model_config, lora_config_dict, backup_adapters),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", (2,))
@pytest.mark.parametrize("strategy", ("fsdp", "fsdp2"))
@pytest.mark.parametrize("backup_adapters", (True, False))
def test_merged_lora_context_gptoss(world_size, strategy, backup_adapters, tmp_path):
    """Test merged_lora_context with FSDP on GPT-OSS model."""
    rendezvous_file = str(tmp_path / f"rdzv_file_gptoss_{backup_adapters}")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    # Create GPT-OSS model config
    model_config = GptOssConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
    )

    # Create LoRA config for GPT-OSS
    lora_config_dict = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": "all-linear",
        "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        "exclude_modules": ["mlp.router"],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    mp.spawn(
        fn=_test_merged_lora_context_worker,
        args=(world_size, rendezvous_file, strategy, model_config, lora_config_dict, backup_adapters),
        nprocs=world_size,
        join=True,
    )
