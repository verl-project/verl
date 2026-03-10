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
"""Tests for PerLayerGPUOptimizerStep with FSDP2.

Tests correctness of per-layer GPU optimizer step against baseline CPU/GPU optimizer.step().
Uses mp.spawn pattern from test_activation_offload.py.
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import init_device_mesh
from transformers import AutoModelForCausalLM, Qwen2Config

from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    PerLayerGPUOptimizerStep,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
)


def _create_tiny_config():
    """Create a tiny Qwen2 config for fast testing."""
    return Qwen2Config(
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        max_position_embeddings=128,
    )


def _setup_distributed(rank, world_size, rendezvous_file):
    """Initialize distributed process group."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    return init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))


def _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False):
    """Create model with FSDP2 wrapping."""
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        model = model.to(device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
    )
    cpu_offload = CPUOffloadPolicy(pin_memory=True) if use_cpu_offload else None
    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": mp_policy,
        "offload_policy": cpu_offload,
    }
    apply_fsdp2(model, fsdp_kwargs, {})
    return model


def _run_forward_backward(model, config, rank):
    """Run a forward+backward pass and return loss value."""
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=f"cuda:{rank}")
    loss = model(input_ids=input_ids).logits.mean()
    loss.backward()
    return loss.item()


# =============================================================================
# Test 1: Layer grouping
# =============================================================================
def _test_layer_grouping_worker(rank, world_size, rendezvous_file):
    device_mesh = _setup_distributed(rank, world_size, rendezvous_file)
    config = _create_tiny_config()
    model = _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    stepper = PerLayerGPUOptimizerStep(model, optimizer, device_id=rank)
    groups = stepper._layer_param_groups

    # Should have num_hidden_layers groups (transformer layers) + 1 residual group (lm_head, final norm)
    assert len(groups) >= config.num_hidden_layers, (
        f"Expected at least {config.num_hidden_layers} groups, got {len(groups)}"
    )

    # Verify all trainable params are covered
    all_params_in_groups = set()
    for g in groups:
        for p in g:
            all_params_in_groups.add(id(p))

    model_params = set(id(p) for p in model.parameters() if p.requires_grad)
    assert all_params_in_groups == model_params, (
        f"Missing params: {model_params - all_params_in_groups}"
    )

    if rank == 0:
        print(f"Layer grouping test passed: {len(groups)} groups, {len(model_params)} params")

    dist.destroy_process_group()


# =============================================================================
# Test 2: CPUOffloadPolicy raises ValueError
# =============================================================================
def _test_cpuoffload_raises_error_worker(rank, world_size, rendezvous_file):
    device_mesh = _setup_distributed(rank, world_size, rendezvous_file)
    config = _create_tiny_config()

    model = _create_model_and_fsdp(config, device_mesh, use_cpu_offload=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # PerLayerGPUOptimizerStep requires params on GPU; CPUOffloadPolicy puts them on CPU
    raised = False
    try:
        PerLayerGPUOptimizerStep(model, optimizer, device_id=rank)
    except ValueError as e:
        raised = True
        if rank == 0:
            print(f"Correctly raised ValueError: {e}")

    assert raised, "Expected ValueError when using CPUOffloadPolicy with PerLayerGPUOptimizerStep"

    del model, optimizer
    dist.destroy_process_group()


# =============================================================================
# Test 3: Correctness - GPU baseline (no offload)
# =============================================================================
def _test_correctness_gpu_baseline_worker(rank, world_size, rendezvous_file):
    device_mesh = _setup_distributed(rank, world_size, rendezvous_file)
    config = _create_tiny_config()

    # --- Baseline: standard optimizer.step() on GPU ---
    torch.manual_seed(42)
    model_baseline = _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False)
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=1e-3)

    torch.manual_seed(100 + rank)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=f"cuda:{rank}")

    loss = model_baseline(input_ids=input_ids).logits.mean()
    loss.backward()
    fsdp2_clip_grad_norm_(model_baseline.parameters(), max_norm=1.0)
    optimizer_baseline.step()
    optimizer_baseline.zero_grad()

    baseline_params = {}
    for n, p in model_baseline.named_parameters():
        local = p._local_tensor if hasattr(p, "_local_tensor") else p
        baseline_params[n] = local.detach().float().cpu().clone()

    del model_baseline, optimizer_baseline
    torch.cuda.empty_cache()

    # --- Per-layer GPU optimizer step (params already on GPU, only optimizer states streamed) ---
    torch.manual_seed(42)
    model_perlayer = _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False)
    optimizer_perlayer = torch.optim.AdamW(model_perlayer.parameters(), lr=1e-3)

    torch.manual_seed(100 + rank)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=f"cuda:{rank}")

    loss = model_perlayer(input_ids=input_ids).logits.mean()
    loss.backward()
    fsdp2_clip_grad_norm_(model_perlayer.parameters(), max_norm=1.0)

    # First, offload optimizer states to CPU (to simulate param_offload + optimizer_offload mode)
    from verl.utils.fsdp_utils import offload_fsdp_optimizer
    offload_fsdp_optimizer(optimizer_perlayer)
    torch.cuda.synchronize()

    stepper = PerLayerGPUOptimizerStep(model_perlayer, optimizer_perlayer, device_id=rank, prefetch_layers=1)
    stepper.step()
    optimizer_perlayer.zero_grad()

    # Compare
    max_diff = 0.0
    for n, p in model_perlayer.named_parameters():
        local = p._local_tensor if hasattr(p, "_local_tensor") else p
        perlayer_val = local.detach().float().cpu()
        diff = (perlayer_val - baseline_params[n]).abs().max().item()
        max_diff = max(max_diff, diff)
        assert diff < 1e-4, f"Param {n} diff={diff:.6e} exceeds threshold"

    if rank == 0:
        print(f"GPU baseline correctness test passed. Max diff: {max_diff:.6e}")

    del model_perlayer, optimizer_perlayer
    dist.destroy_process_group()


# =============================================================================
# Test 4: Multi-step stability
# =============================================================================
def _test_multi_step_worker(rank, world_size, rendezvous_file):
    device_mesh = _setup_distributed(rank, world_size, rendezvous_file)
    config = _create_tiny_config()

    model = _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Offload optimizer states to CPU (simulates optimizer_offload=True)
    from verl.utils.fsdp_utils import offload_fsdp_optimizer
    offload_fsdp_optimizer(optimizer)
    torch.cuda.synchronize()

    # Create stepper once and reuse across steps (same as production code)
    stepper = PerLayerGPUOptimizerStep(model, optimizer, device_id=rank, prefetch_layers=1)

    losses = []
    for step in range(3):
        torch.manual_seed(step * 100 + rank)
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device=f"cuda:{rank}")
        loss = model(input_ids=input_ids).logits.mean()
        loss.backward()
        fsdp2_clip_grad_norm_(model.parameters(), max_norm=1.0)

        stepper.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    if rank == 0:
        print(f"Multi-step test passed. Losses: {losses}")

    dist.destroy_process_group()


# =============================================================================
# Test 5: Different prefetch_layers values
# =============================================================================
def _test_prefetch_layers_worker(rank, world_size, rendezvous_file, prefetch_layers):
    device_mesh = _setup_distributed(rank, world_size, rendezvous_file)
    config = _create_tiny_config()

    model = _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Offload optimizer states to CPU
    from verl.utils.fsdp_utils import offload_fsdp_optimizer
    offload_fsdp_optimizer(optimizer)
    torch.cuda.synchronize()

    torch.manual_seed(42 + rank)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=f"cuda:{rank}")
    loss = model(input_ids=input_ids).logits.mean()
    loss.backward()
    fsdp2_clip_grad_norm_(model.parameters(), max_norm=1.0)

    stepper = PerLayerGPUOptimizerStep(model, optimizer, device_id=rank, prefetch_layers=prefetch_layers)
    stepper.step()
    optimizer.zero_grad()

    # Verify params were updated (not all zeros)
    updated = False
    for n, p in model.named_parameters():
        local = p._local_tensor if hasattr(p, "_local_tensor") else p
        if local.abs().sum() > 0:
            updated = True
            break
    assert updated, "No params were updated"

    if rank == 0:
        print(f"Prefetch layers={prefetch_layers} test passed")

    dist.destroy_process_group()


# =============================================================================
# Pytest entry points
# =============================================================================
@pytest.mark.parametrize("world_size", [2])
def test_layer_grouping(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_file")
    mp.spawn(
        fn=_test_layer_grouping_worker,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", [2])
def test_cpuoffload_raises_error(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_file")
    mp.spawn(
        fn=_test_cpuoffload_raises_error_worker,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", [2])
def test_correctness_gpu_baseline(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_file")
    mp.spawn(
        fn=_test_correctness_gpu_baseline_worker,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", [2])
def test_multi_step(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_file")
    mp.spawn(
        fn=_test_multi_step_worker,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size,prefetch_layers", [(2, 0), (2, 1), (2, 2)])
def test_prefetch_layers(world_size, prefetch_layers, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_file")
    mp.spawn(
        fn=_test_prefetch_layers_worker,
        args=(world_size, rendezvous_file, prefetch_layers),
        nprocs=world_size,
        join=True,
    )
