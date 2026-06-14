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
"""
Test script to verify TiledMLP accuracy by comparing logits and gradients
between regular MLP and TiledMLP under FSDP2.
Run with: torchrun --nproc_per_node=2 tests/test_tiled_mlp_accuracy.py
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def create_model(model_name="Qwen/Qwen3-1.7B", num_layers=2):
    """Load a Qwen3-1.7B model with only 2 layers from pretrained weights."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_hidden_layers = num_layers

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    return model


def apply_fsdp2(model, device_mesh):
    """Apply FSDP2 sharding to model."""
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh)
    fully_shard(model, mesh=device_mesh)
    return model


def run_forward_backward(model, input_ids, labels):
    """Run forward and backward pass, return logits and gradients."""
    model.zero_grad()

    outputs = model(input_ids=input_ids, labels=labels)
    logits = outputs.logits.clone().detach()
    loss = outputs.loss

    loss.backward()

    # Collect MLP gradients
    gradients = {}
    for name, param in model.named_parameters():
        if "mlp" in name and param.grad is not None:
            gradients[name] = param.grad.clone().detach()

    return logits, gradients, loss.item()


def compare_results(logits1, grads1, logits2, grads2, rank):
    """Compare logits and gradients between two runs."""
    # Compare logits
    logits_diff = (logits1 - logits2).abs()
    logits_max_diff = logits_diff.max().item()
    logits_mean_diff = logits_diff.mean().item()

    # Compare gradients (only for params that exist on this rank due to FSDP sharding)
    all_pass = True
    grad_results = []
    for name in sorted(grads1.keys()):
        if name in grads2:
            g1, g2 = grads1[name], grads2[name]
            diff = (g1 - g2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Check if within tolerance (1e-2 for bf16)
            passed = max_diff < 1e-2
            if not passed:
                all_pass = False
            grad_results.append((name, max_diff, mean_diff, passed))

    # Only print on rank 0 to avoid duplicate output
    if rank == 0:
        print("\n=== Comparison Results ===")
        print("\nLogits:")
        print(f"  Max diff: {logits_max_diff:.2e}")
        print(f"  Mean diff: {logits_mean_diff:.2e}")

        print("\nMLP Parameter Gradients:")
        if grad_results:
            for name, max_diff, mean_diff, passed in grad_results:
                status = "✓" if passed else "✗"
                print(f"  {name}: max={max_diff:.2e}, mean={mean_diff:.2e} {status}")
        else:
            print("  (Gradients sharded to other ranks under FSDP2)")

    return all_pass


def main():
    rank, world_size = setup_distributed()
    device_mesh = init_device_mesh("cuda", (world_size,))

    model_name = "Qwen/Qwen3-1.7B"
    num_layers = 2

    if rank == 0:
        print(f"Running TiledMLP accuracy test with {world_size} GPUs")
        print(f"Model: {model_name} ({num_layers} layers, from pretrained)")

    dist.barrier()

    # ========== Create Model 1: WITHOUT TiledMLP ==========
    if rank == 0:
        print("\n" + "=" * 60)
        print("Creating Model 1 (without TiledMLP)")
        print("=" * 60)

    model1 = create_model(model_name, num_layers)
    model1 = apply_fsdp2(model1, device_mesh)
    model1 = model1.cuda()

    # Create deterministic input
    torch.manual_seed(42)
    batch_size, seq_len = 2, 256
    vocab_size = model1.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    labels = input_ids.clone()

    # ========== Run Model 1: WITHOUT TiledMLP ==========
    if rank == 0:
        print("\n" + "=" * 60)
        print("Running forward/backward on Model 1 (without TiledMLP)")
        print("=" * 60)

    logits1, grads1, loss1 = run_forward_backward(model1, input_ids, labels)
    if rank == 0:
        print(f"Loss: {loss1:.4f}")

    # Free model1 memory before creating model2
    del model1
    torch.cuda.empty_cache()

    dist.barrier()

    # ========== Create Model 2, apply TiledMLP patch, then FSDP2 ==========
    if rank == 0:
        print("\n" + "=" * 60)
        print("Creating Model 2 (with TiledMLP, patch before FSDP2)")
        print("=" * 60)

    model2 = create_model(model_name, num_layers)

    # Apply TiledMLP patch AFTER model instantiation but BEFORE FSDP2 wrap
    if rank == 0:
        print("Applying TiledMLP monkey patch before FSDP2...")

    from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch

    apply_tiled_mlp_monkey_patch(num_shards=4, model_type="qwen3")

    model2 = apply_fsdp2(model2, device_mesh)
    model2 = model2.cuda()

    dist.barrier()

    # ========== Run Model 2: WITH TiledMLP ==========
    if rank == 0:
        print("\n" + "=" * 60)
        print("Running forward/backward on Model 2 (with TiledMLP)")
        print("=" * 60)

    logits2, grads2, loss2 = run_forward_backward(model2, input_ids, labels)
    if rank == 0:
        print(f"Loss: {loss2:.4f}")

    dist.barrier()

    # ========== Compare Results ==========
    all_pass = compare_results(logits1, grads1, logits2, grads2, rank)

    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Loss diff: {abs(loss1 - loss2):.2e}")
        print(f"All gradient checks: {'PASS' if all_pass else 'FAIL'}")

    # Cleanup
    del model2
    torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
