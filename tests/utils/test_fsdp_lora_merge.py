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
    collect_merged_lora_params,
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
            config=model_config, torch_dtype=torch.bfloat16, attn_implementation="eager"
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


def _test_collect_merged_lora_params_worker(rank, world_size, rendezvous_file, strategy, lora_targets):
    """Worker function for testing collect_merged_lora_params."""
    get_torch_device().set_device(rank)
    torch.distributed.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    device_mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

    model_config = Qwen2Config(num_hidden_layers=2, num_attention_heads=2, hidden_size=128)
    with torch.device(get_device_name()):
        model = AutoModelForCausalLM.from_config(
            config=model_config, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        model = model.to(device=get_device_name())

    # Get base model keys before adding LoRA
    base_keys = set(model.state_dict().keys())

    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=lora_targets, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Initialize LoRA weights to non-zero values
    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                for adapter_name in module.lora_A.keys():
                    module.lora_A[adapter_name].weight.data.uniform_(0.5, 1.5)
                    module.lora_B[adapter_name].weight.data.uniform_(1.5, 2.5)

    from peft.utils.save_and_load import get_peft_model_state_dict

    # Wrap with FSDP
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
        fsdp_kwargs = {"mesh": device_mesh, "mp_policy": mp_policy}
        apply_fsdp2(model, fsdp_kwargs, {})

    # Save adapter weights AFTER FSDP wrapping so dtypes match (bfloat16 mixed precision).
    original_adapter_weights = {}
    for key, val in get_peft_model_state_dict(model).items():
        if hasattr(val, "full_tensor"):
            val = val.full_tensor()
        original_adapter_weights[key] = val.detach().cpu().clone()

    # Call collect_merged_lora_params
    merged_params = collect_merged_lora_params(model)

    # 1. All returned tensors should be on CPU
    for key, val in merged_params.items():
        assert val.is_cpu, f"Expected CPU tensor for {key}, got {val.device}"

    # 2. No LoRA or FSDP artifact keys
    for key in merged_params.keys():
        assert "lora_" not in key, f"LoRA key should be filtered: {key}"
        assert "_flat_param" not in key, f"FSDP flat param should be filtered: {key}"
        assert "_fsdp_wrapped_module" not in key, f"FSDP wrapper prefix should be stripped: {key}"
        assert ".base_layer" not in key, f"peft base_layer should be stripped: {key}"

    # 3. Keys should match base model format (model.layers.X.self_attn.q_proj.weight etc.)
    for base_key in base_keys:
        assert base_key in merged_params, f"Base model key {base_key} missing from merged params"

    # 4. Merged params should have the same count as base model (no extras)
    assert len(merged_params) == len(base_keys), (
        f"Expected {len(base_keys)} params, got {len(merged_params)}. "
        f"Extra keys: {set(merged_params.keys()) - base_keys}"
    )

    # 5. LoRA should be unmerged after extraction (training state restored)
    lora_layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    for layer in lora_layers:
        assert not getattr(layer, "merged", False), "LoRA should be unmerged after collect_merged_lora_params"

    # 6. Adapter weights should be preserved
    # For FSDP2, state_dict() returns sharded DTensors; unshard via full_tensor() before comparing.
    restored_adapter_weights = get_peft_model_state_dict(model)
    for key in original_adapter_weights.keys():
        assert key in restored_adapter_weights, f"Adapter key {key} missing after extraction"
        restored = restored_adapter_weights[key]
        if hasattr(restored, "full_tensor"):
            restored = restored.full_tensor()
        torch.testing.assert_close(
            original_adapter_weights[key].cpu(),
            restored.cpu(),
            rtol=1e-5,
            atol=1e-6,
            msg=f"Adapter weight {key} changed after extraction",
        )

    # 7. Calling twice should produce identical results (idempotent)
    merged_params_2 = collect_merged_lora_params(model)
    assert set(merged_params.keys()) == set(merged_params_2.keys()), "Keys should be identical across calls"
    for key in merged_params.keys():
        torch.testing.assert_close(
            merged_params[key], merged_params_2[key], rtol=1e-5, atol=1e-6, msg=f"Values differ for {key}"
        )

    # 8. Value correctness: merged weights should equal base + LoRA delta for
    # targeted modules, and match base weights exactly for non-targeted modules.
    # Build a non-FSDP reference by merging LoRA on an unwrapped copy.
    ref_config = Qwen2Config(num_hidden_layers=2, num_attention_heads=2, hidden_size=128)
    with torch.device("cpu"):
        ref_model = AutoModelForCausalLM.from_config(
            config=ref_config, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
    ref_lora = LoraConfig(
        r=8, lora_alpha=16, target_modules=lora_targets, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"
    )
    ref_model = get_peft_model(ref_model, ref_lora)
    # Copy the same base + adapter weights from the FSDP model
    fsdp_full_sd = {}
    if strategy == "fsdp":
        with FSDP.summon_full_params(model, writeback=False):
            for k, v in model.state_dict().items():
                clean = k.replace("_fsdp_wrapped_module.", "")
                fsdp_full_sd[clean] = v.full_tensor().cpu() if hasattr(v, "full_tensor") else v.cpu()
    else:
        with FSDP.summon_full_params(model, writeback=False):
            for k, v in model.state_dict().items():
                fsdp_full_sd[k] = v.full_tensor().cpu() if hasattr(v, "full_tensor") else v.cpu()
    ref_model.load_state_dict(fsdp_full_sd, strict=False)
    # Move to GPU before merging so peft uses the same bfloat16 matmul path
    # as collect_merged_lora_params (peft uses float32 on CPU, bfloat16 on GPU)
    ref_model = ref_model.to(device=get_device_name())

    ref_merged = ref_model.merge_and_unload()
    ref_sd = ref_merged.state_dict()
    # Tolerance accounts for bfloat16 matmul non-determinism: FSDP all-gathered
    # tensors may have different memory layouts than contiguous tensors, causing
    # cuBLAS to pick different kernels with up to 1-2 ULP difference.
    # At magnitude ~32, bfloat16 ULP = 32 * 2^(-7) = 0.25.
    for key in base_keys:
        assert key in ref_sd, f"Reference missing {key}"
        torch.testing.assert_close(
            merged_params[key].to(torch.float32),
            ref_sd[key].to(torch.float32).cpu(),
            rtol=1e-2,
            atol=0.5,
            msg=f"Value mismatch for {key}",
        )

    if rank == 0:
        print(f"collect_merged_lora_params test with {strategy} passed on {world_size} GPUs!")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2,))
@pytest.mark.parametrize("strategy", ("fsdp", "fsdp2"))
@pytest.mark.parametrize("lora_targets", (["q_proj", "v_proj"], "all-linear"))
def test_collect_merged_lora_params(world_size, strategy, lora_targets, tmp_path):
    """Test collect_merged_lora_params extracts correct HF-format merged weights."""
    suffix = "alllinear" if lora_targets == "all-linear" else "qv"
    rendezvous_file = str(tmp_path / f"rdzv_file_merged_params_{strategy}_{suffix}")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_test_collect_merged_lora_params_worker,
        args=(world_size, rendezvous_file, strategy, lora_targets),
        nprocs=world_size,
        join=True,
    )
