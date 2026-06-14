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
FSDP2-compatible TiledMLP implementation for memory-efficient MLP computation.

This module provides a tiled MLP implementation that reduces peak memory usage
by processing the MLP forward/backward pass in chunks (tiles). This is particularly
useful for large models with FSDP2 training.
"""

import threading
from typing import Optional

import torch
import torch.nn as nn


class GradientAccumulator:
    """Gradient accumulator for TiledMLP (FSDP compatible).

    This class manages gradient accumulation across multiple shards during
    the backward pass of TiledMLP. It ensures correct gradient computation
    when processing input in chunks.
    """

    def __init__(self, params: list[torch.nn.Parameter], total_shards: int, dtype: torch.dtype = None):
        self.params = params
        self.total_shards = total_shards
        self.grad_accumulation_dtype = dtype or torch.float32
        self.accumulated_grads = {}
        self.hooks = []
        self.lock = threading.Lock()

        for param in self.params:
            if param.grad is not None:
                self.accumulated_grads[param] = param.grad.to(self.grad_accumulation_dtype)
                param.grad = None
            else:
                self.accumulated_grads[param] = torch.zeros_like(param, dtype=self.grad_accumulation_dtype)

    def install_hooks(self, is_last_shard: bool):
        """Install gradient hooks for the current shard."""
        self._remove_hooks()

        def create_hook(param):
            def hook(grad):
                with self.lock:
                    grad_to_accum_dtype = grad.to(self.grad_accumulation_dtype)
                    self.accumulated_grads[param] += grad_to_accum_dtype

                    if is_last_shard:
                        param.grad = None  # Critical: prevent double accumulation
                        final_grad = self.accumulated_grads[param].to(param.dtype)
                        return final_grad
                    return None

            return hook

        for param in self.params:
            if param.requires_grad:
                hook = param.register_hook(create_hook(param))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def cleanup(self):
        """Cleanup hooks and resources."""
        self._remove_hooks()


class TiledMLP(torch.autograd.Function):
    """TiledMLP implementation for memory-efficient MLP computation.

    This autograd function processes MLP forward/backward in tiles (chunks)
    to reduce peak memory usage. Compatible with FSDP2.
    """

    @staticmethod
    def forward(ctx, fn, module, x, shards, compute_params):
        ctx.fn = fn
        ctx.module = module
        ctx.shards = shards
        ctx.compute_params = [p for p in compute_params if p.requires_grad]
        ctx.save_for_backward(x)

        # Split on dim=-2 (seqlen dimension) following Liger Kernel style
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(module, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)
        return output_unsharded

    @staticmethod
    def backward(ctx, *grads):
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        module = ctx.module
        shards = ctx.shards
        compute_params = ctx.compute_params

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)

        # Flatten to [bs*seqlen, hidden_size]
        hidden_size = x.shape[-1]
        x_shape_orig = x.shape
        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)

        # Pre-allocate input gradient
        x_grad = torch.zeros_like(x)

        # Split on dim=0
        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        grad_accumulator = GradientAccumulator(compute_params, shards, dtype=x.dtype)

        for i, x_shard in enumerate(x_shards):
            x_shard.requires_grad_(x_requires_grad)

            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            # narrow(0, ...) creates a contiguous view that can receive gradients
            x_shard.grad = x_grad.narrow(0, shard_offset, shard_step)
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step)

            is_last_shard = i + 1 == shards
            grad_accumulator.install_hooks(is_last_shard)

            with torch.enable_grad():
                output = fn(module, x_shard)
            torch.autograd.backward(output, incoming_grad_shard)

        grad_accumulator.cleanup()
        del grad_accumulator

        # Restore original shape
        x_grad = x_grad.view(x_shape_orig) if x_requires_grad else None
        return (None, None, x_grad, None, None)


def _mlp_forward_fn(module, x):
    """Forward function for LlamaMLP / Qwen2MLP / Qwen3MLP style."""
    return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))


# ============================================================================
# Monkey Patch Functions
# ============================================================================

# Model type to MLP class mapping
_MODEL_TYPE_TO_MLP_CLASS = {
    "llama": ("transformers.models.llama.modeling_llama", "LlamaMLP"),
    "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),
    "qwen2_5": ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),  # Qwen2.5 uses Qwen2 MLP
    "qwen3": ("transformers.models.qwen3.modeling_qwen3", "Qwen3MLP"),
}


def apply_tiled_mlp_monkey_patch(
    num_shards: int = 4,
    model_type: Optional[str] = None,
):
    """Apply TiledMLP monkey patch based on model_type.

    This function MUST be called BEFORE model instantiation to take effect.
    It patches the MLP classes in transformers library to use TiledMLP for
    memory-efficient computation during training.

    Args:
        num_shards: Number of shards to split the input into. Higher values
                   reduce peak memory but may slightly impact performance.
        model_type: The model type string (e.g., "llama", "qwen2", "qwen3").
                   If None, patches all supported model types.

    Returns:
        List of patched class names.
    """
    if model_type is None:
        types_to_patch = list(_MODEL_TYPE_TO_MLP_CLASS.keys())
    elif model_type in _MODEL_TYPE_TO_MLP_CLASS:
        types_to_patch = [model_type]
    else:
        raise ValueError(
            f"TiledMLP does not support model_type='{model_type}'. "
            f"Supported types: {list(_MODEL_TYPE_TO_MLP_CLASS.keys())}. "
            f"For SwiGLU-style MLPs, you can add support by extending _MODEL_TYPE_TO_MLP_CLASS "
            f"in verl/models/transformers/tiled_mlp.py"
        )

    patched_classes = []

    for mtype in types_to_patch:
        module_path, class_name = _MODEL_TYPE_TO_MLP_CLASS[mtype]
        try:
            import importlib

            module = importlib.import_module(module_path)
            mlp_class = getattr(module, class_name)
            _patch_mlp_class(mlp_class, _mlp_forward_fn, num_shards)
            if class_name not in patched_classes:
                patched_classes.append(class_name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not patch {mtype} MLP: {e}")

    if patched_classes:
        print(f"TiledMLP monkey patch applied to: {', '.join(patched_classes)} (shards={num_shards})")

    return patched_classes


def _patch_mlp_class(mlp_class: type[nn.Module], forward_fn, num_shards: int):
    """Patch a single MLP class to use TiledMLP."""

    def tiled_forward(self, x):
        compute_params = [p for p in self.parameters() if p.requires_grad]
        return TiledMLP.apply(forward_fn, self, x, num_shards, compute_params)

    mlp_class.forward = tiled_forward
