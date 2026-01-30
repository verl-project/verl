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
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from transformers import AutoModelForCausalLM, Qwen3Config

from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.fsdp_utils import (
    MixedPrecisionPolicy,
    apply_fsdp2,
    get_fsdp_wrap_policy,
    normalize_peft_param_name,
)
from verl.utils.model import convert_weight_keys


def _test_normalize_peft_with_fsdp_worker(rank, world_size, rendezvous_file, strategy):
    """Worker function for testing normalize_peft_param_name with FSDP-wrapped models.

    Args:
        rank: Process rank
        world_size: Total number of processes
        rendezvous_file: Path to rendezvous file for distributed init
        strategy: FSDP strategy ("fsdp" or "fsdp2")
    """
    get_torch_device().set_device(rank)
    torch.distributed.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    device_mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

    # Create model config
    config = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
    )

    # Create base model
    with torch.device(get_device_name()):
        base_model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        base_model = base_model.to(device=get_device_name())

    # Create PEFT model with LoRA
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(base_model, lora_config)

    # Wrap base model with FSDP (create a fresh copy for base model)
    with torch.device(get_device_name()):
        base_model_for_fsdp = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        base_model_for_fsdp = base_model_for_fsdp.to(device=get_device_name())

    if strategy == "fsdp":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        # Wrap base model with FSDP
        fsdp_base_model = FSDP(
            base_model_for_fsdp,
            use_orig_params=True,
            device_id=get_torch_device().current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
            auto_wrap_policy=get_fsdp_wrap_policy(module=base_model_for_fsdp, is_lora=False),
        )

        # Wrap PEFT model with FSDP
        fsdp_peft_model = FSDP(
            peft_model,
            use_orig_params=True,
            device_id=get_torch_device().current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
            auto_wrap_policy=get_fsdp_wrap_policy(module=peft_model, is_lora=True),
        )
    else:
        # FSDP2
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        )
        fsdp_kwargs = {
            "mesh": device_mesh,
            "mp_policy": mp_policy,
        }

        # Wrap base model with FSDP2
        apply_fsdp2(base_model_for_fsdp, fsdp_kwargs, {})
        fsdp_base_model = base_model_for_fsdp

        # Wrap PEFT model with FSDP2
        apply_fsdp2(peft_model, fsdp_kwargs, {})
        fsdp_peft_model = peft_model

    # Get state dicts from FSDP models
    if strategy == "fsdp":
        # FSDP v1: Use full_state_dict context
        with FSDP.state_dict_type(fsdp_base_model, StateDictType.FULL_STATE_DICT):
            base_state_dict = fsdp_base_model.state_dict()

        with FSDP.state_dict_type(fsdp_peft_model, StateDictType.FULL_STATE_DICT):
            peft_state_dict = fsdp_peft_model.state_dict()
    else:
        # FSDP2: Direct state_dict call
        base_state_dict = fsdp_base_model.state_dict()
        peft_state_dict = fsdp_peft_model.state_dict()

    # Normalize PEFT model state dict
    normalized_peft_state_dict = normalize_peft_param_name(peft_state_dict)

    base_state_dict = convert_weight_keys(
        base_state_dict, getattr(fsdp_base_model, "_fsdp_wrapped_module", fsdp_base_model)
    )
    normalized_peft_state_dict = convert_weight_keys(
        normalized_peft_state_dict, getattr(fsdp_peft_model, "_fsdp_wrapped_module", fsdp_peft_model)
    )

    # Get key sets
    base_keys = set(base_state_dict.keys())
    normalized_peft_keys = set(normalized_peft_state_dict.keys())

    # if rank == 0:
    print(f"\n=== FSDP {strategy} Test Results ===")
    print(f"Base model keys: {base_keys=}")
    print(f"Normalized PEFT keys: {normalized_peft_keys=}")

    # Check for missing keys
    missing_keys = base_keys - normalized_peft_keys
    if missing_keys:
        print(f"Missing keys from base model: {missing_keys}")

    # Check for extra keys
    extra_keys = normalized_peft_keys - base_keys
    if extra_keys:
        print(f"Extra keys not in base model: {extra_keys}")

    # Verify that all base model keys are in the normalized PEFT keys
    missing_keys = base_keys - normalized_peft_keys
    assert len(missing_keys) == 0, f"Missing keys from base model: {missing_keys}"

    # Verify that all normalized PEFT keys are in the base model
    extra_keys = normalized_peft_keys - base_keys
    assert len(extra_keys) == 0, f"Extra keys not in base model: {extra_keys}"

    # Verify exact match
    assert base_keys == normalized_peft_keys, "Normalized PEFT keys should exactly match FSDP base model keys"

    # Verify tensor shapes match
    for key in base_keys:
        base_shape = base_state_dict[key].shape
        peft_shape = normalized_peft_state_dict[key].shape
        assert base_shape == peft_shape, f"Shape mismatch for {key}: base={base_shape}, peft={peft_shape}"

    # Verify no LoRA keys remain in normalized state dict
    lora_keys = [k for k in normalized_peft_keys if "lora_" in k or "adapter_" in k]
    assert len(lora_keys) == 0, f"Normalized state dict should not contain LoRA keys, but found: {lora_keys}"

    if rank == 0:
        print(f"✓ All tests passed for FSDP {strategy}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2,))
@pytest.mark.parametrize("strategy", ("fsdp", "fsdp2"))
def test_normalize_peft_param_name_with_fsdp(world_size, strategy, tmp_path):
    """Test normalize_peft_param_name with FSDP-wrapped models.

    This test verifies that after applying FSDP to both base and PEFT models,
    the normalized PEFT model keys match the FSDP base model keys.
    """
    rendezvous_file = str(tmp_path / f"rdzv_file_normalize_{strategy}")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_test_normalize_peft_with_fsdp_worker,
        args=(world_size, rendezvous_file, strategy),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
