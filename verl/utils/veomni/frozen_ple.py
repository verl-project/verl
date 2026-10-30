"""Freeze Qwen4-Exp PLE tables before optimizer construction."""

import re

import torch

_PLE_TABLE = re.compile(r"(?:^|\.)ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")


def is_ple_table(name: str) -> bool:
    return _PLE_TABLE.search(name) is not None


def validate_frozen_ple_rollout(freeze_ple_embeddings: bool, rollout_name: str, load_format: str) -> None:
    """Require real initial PLE weights when frozen tables are excluded from sync.

    The NPU hybrid rollout preserves its weights across level-1 sleep. Frozen
    tables can therefore be loaded once from the actor's model checkpoint.
    Dummy initialization never reads those tables and subsequent actor updates
    deliberately do not transmit them.
    """
    if freeze_ple_embeddings and rollout_name == "vllm" and load_format not in ("auto", "safetensors"):
        raise ValueError(
            "freeze_ple_embeddings excludes PLE tables from actor-to-rollout weight sync; "
            f"rollout.load_format={load_format!r} does not guarantee checkpoint initialization. "
            "Use actor_rollout_ref.rollout.load_format=safetensors (or auto) so rollout "
            "loads the same immutable PLE tables from model.path before the first weight sync."
        )


def prepare_ple_storage(model: torch.nn.Module, dtype: str | None) -> None:
    """Set table storage before freezing; leave all trainable weights unchanged."""
    if dtype is None:
        return
    if dtype not in ("float32", "bfloat16"):
        raise ValueError(f"Unsupported frozen PLE dtype: {dtype}")
    for name, param in model.named_parameters():
        if is_ple_table(name):
            # DTensor .data assignment changes wrapper dtype but leaves its
            # local shard unchanged. Swap the complete tensor implementation.
            converted = torch.nn.Parameter(param.to(dtype=getattr(torch, dtype)), requires_grad=param.requires_grad)
            torch.utils.swap_tensors(param, converted)


def freeze_ple_tables(model: torch.nn.Module) -> tuple[int, int]:
    """Exclude only PLE lookup tables from autograd and optimizer construction."""
    if getattr(model.config, "model_type", None) != "qwen4_exp":
        raise ValueError("freeze_ple_embeddings currently supports only qwen4_exp")
    count = local_bytes = 0
    for name, param in model.named_parameters():
        if not is_ple_table(name):
            continue
        param.requires_grad_(False)
        param.grad = None
        local = param.to_local() if hasattr(param, "to_local") else param
        count += 1
        local_bytes += local.numel() * local.element_size()
    if not count:
        raise ValueError("freeze_ple_embeddings requested but no PLE table parameters were found")
    return count, local_bytes
