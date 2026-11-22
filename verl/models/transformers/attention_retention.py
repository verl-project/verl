# Copyright contributors to the PEAR project
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

import importlib
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps

import torch
from torch.utils.checkpoint import checkpoint

_active_cache: ContextVar = ContextVar("attention_checkpoint_cache", default=None)


# the forward that the cache wraps, by attention implementation: (module, function)
_FLASH_ATTENTION_FORWARDS = {
    "flash_attention_4": ("flash_attn.cute.interface", "_flash_attn_fwd"),
    "flash_attention_3": ("flash_attn_interface", "_flash_attn_forward"),  # Hopper
}


def _install_flash_attention_cache(module_name: str, function_name: str):
    """Wraps a FlashAttention forward: inside a retaining checkpoint the recompute replays the cached outputs."""
    module = importlib.import_module(module_name)
    original = getattr(module, function_name)
    if getattr(original, "_retains_outputs", False):
        return

    @wraps(original)
    def forward(*args, **kwargs):
        active = _active_cache.get()
        if active is None:
            return original(*args, **kwargs)
        cache, recompute = active
        if recompute:
            return cache.pop()
        outputs = original(*args, **kwargs)
        cache.appendleft(tuple(t.detach() if isinstance(t, torch.Tensor) else t for t in outputs))
        return outputs

    forward._retains_outputs = True
    setattr(module, function_name, forward)


def _checkpoint_contexts():
    cache = deque()

    @contextmanager
    def context(recompute: bool):
        token = _active_cache.set((cache, recompute))
        try:
            yield
        finally:
            _active_cache.reset(token)
            if recompute:
                cache.clear()

    return context(False), context(True)


def _attn_implementations(model: torch.nn.Module) -> set:
    """The attention implementations of the model's config and of its sub-models' configs."""
    configs = [model.config] + [getattr(model.config, name, None) for name in getattr(model.config, "sub_configs", {})]
    return {getattr(config, "_attn_implementation", None) for config in configs}


def retain_attention_in_checkpoints(model: torch.nn.Module):
    """Keeps a checkpointed attention layer's FlashAttention-4 or -3 outputs from its forward to its backward."""
    for name in _attn_implementations(model) & _FLASH_ATTENTION_FORWARDS.keys():
        _install_flash_attention_cache(*_FLASH_ATTENTION_FORWARDS[name])
    for layer in model.modules():
        if not getattr(layer, "gradient_checkpointing", False):
            continue
        # a decoder layer's type is its block_type from transformers 5.16, its layer_type before
        block_type = getattr(layer, "block_type", getattr(layer, "layer_type", "full_attention"))
        if block_type != "full_attention":  # linear attention: nothing to retain
            continue
        original = layer._gradient_checkpointing_func
        # the checkpoint calls context_fn afresh for every forward: a cache lives for one forward and its backward
        options = dict(getattr(original, "keywords", None) or {})
        options.update(use_reentrant=False, context_fn=_checkpoint_contexts)
        function = original.func if isinstance(original, partial) else checkpoint
        arguments = original.args if isinstance(original, partial) else ()
        layer._gradient_checkpointing_func = partial(function, *arguments, **options)
