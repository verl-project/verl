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

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

WeightUpdate = tuple[str, torch.Tensor]


def tied_embedding_aliases(model: torch.nn.Module) -> set[str]:
    """Names of tied word-embedding parameters vLLM skips when loading.

    Mirrors vLLM's own alias detection (``_get_tied_embedding_params``): only
    ``VocabParallelEmbedding`` parameters count, and the first qualname reached
    by traversal is the canonical one every alias defers to.
    """
    canonical: dict[int, str] = {}
    aliases: set[str] = set()
    for prefix, submodule in model.named_modules(remove_duplicate=False):
        if not isinstance(submodule, VocabParallelEmbedding):
            continue
        for name, param in submodule.named_parameters(remove_duplicate=False):
            qualname = f"{prefix}.{name}" if prefix else name
            if canonical.setdefault(id(param), qualname) != qualname:
                aliases.add(qualname)
    return aliases


def drop_tied_alias_updates(model: torch.nn.Module, weights: list[WeightUpdate]) -> list[WeightUpdate]:
    """Drop tied-embedding aliases (e.g. ``lm_head.weight``) from ``weights``.

    ``state_dict()`` on the trainer side lists every qualname of a tied
    parameter, so a refit bucket can carry ``lm_head.weight`` without the
    ``model.embed_tokens.weight`` it aliases. vLLM skips the alias and then
    rejects the bucket because the canonical name never loaded in it. The alias
    is the same storage, so dropping it loses nothing.

    ``aliases`` holds vLLM qualnames, while ``weights`` arrives in checkpoint
    naming, so names go through the model's ``hf_to_vllm_mapper`` first — the
    same order ``AutoWeightsLoader.load_weights`` uses. Without it a wrapped
    model never matches: Qwen3.5-VL sends ``lm_head.weight`` for what vLLM calls
    ``language_model.lm_head.weight``.
    """
    aliases = tied_embedding_aliases(model)
    if not aliases:
        return weights

    mapper = getattr(model, "hf_to_vllm_mapper", None)

    def is_alias(name: str) -> bool:
        if mapper is not None:
            # An empty result means the mapper drops the weight outright; leave
            # that to vLLM rather than treating it as an alias.
            mapped = mapper.apply_list([name])
            name = mapped[0] if mapped else name
        return name in aliases

    return [(name, tensor) for name, tensor in weights if not is_alias(name)]


def split_buffer_updates(
    model: torch.nn.Module, weights: list[WeightUpdate]
) -> tuple[list[WeightUpdate], list[WeightUpdate], dict[str, torch.Tensor]]:
    """Split incoming weight updates into parameter and buffer updates.

    Returns the parameter updates, the buffer updates, and the model's
    ``named_buffers`` map so callers can reuse it without re-iterating.
    """
    named_buffers = dict(model.named_buffers())
    param_updates, buffer_updates = [], []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))
        else:
            param_updates.append((name, tensor))
    return param_updates, buffer_updates, named_buffers


@torch.no_grad()
def apply_buffer_updates(
    model: torch.nn.Module,
    buffer_updates: list[WeightUpdate],
    named_buffers: dict[str, torch.Tensor] | None = None,
) -> int:
    """Copy updated buffer tensors into the target model in-place."""
    if not buffer_updates:
        return 0

    if named_buffers is None:
        named_buffers = dict(model.named_buffers())
    loaded = 0
    for name, tensor in buffer_updates:
        if name not in named_buffers:
            continue

        target = named_buffers[name]
        if target.shape != tensor.shape:
            raise ValueError(
                f"Buffer shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
            )

        source = tensor.to(device=target.device, dtype=target.dtype, non_blocking=False)
        target.copy_(source, non_blocking=False)
        loaded += 1

    return loaded
