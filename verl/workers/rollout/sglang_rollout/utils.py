# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import pickle
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist

from verl.utils.device import get_device_name
from verl.workers.rollout.utils import ensure_async_iterator

SGLANG_LORA_NAME = "verl_actor_lora_name"


def normalize_peft_config_for_sglang(peft_config: dict) -> dict:
    """Normalize an engine's adapter config (enums to strings) for SGLang's adapter loader."""
    normalized = dict(peft_config)
    for key in ("task_type", "peft_type"):
        if key in normalized:
            normalized[key] = getattr(normalized[key], "value", normalized[key])
    if "peft_type" not in normalized:
        raise ValueError(
            "adapter config has no 'peft_type', which SGLang's adapter loader requires. See "
            "BaseEngine.get_per_tensor_param for the keys. Keys present: " + ", ".join(sorted(normalized))
        )
    # A bare string must stay one: list() would tear "all-linear" into characters.
    target_modules = normalized["target_modules"]
    normalized["target_modules"] = target_modules if isinstance(target_modules, str) else list(target_modules)
    return normalized


def lora_rank_of(model_config) -> int:
    """The LoRA rank, from whichever block carries it: megatron sets ``model.lora.rank``, fsdp
    the flat ``model.lora_rank``."""
    return max(int(getattr(model_config, "lora_rank", 0) or 0), int(model_config.lora.get("rank", 0) or 0))


def lora_served_as_adapter(model_config) -> bool:
    """Whether SGLang should serve LoRA as a hot-swappable adapter.

    ``HFModelConfig`` carries two LoRA config blocks that are never synced: megatron
    runs set ``model.lora.rank`` (dict) while fsdp runs set the flat ``model.lora_rank``,
    so both must be checked to detect that LoRA is enabled at all.

    With ``model.lora.merge=True`` the trainer merges the adapter into the base weights
    and pushes a full HF-keyed weight update (``peft_config=None``), so no adapter is ever
    loaded into SGLang: the engine must not be launched with ``enable_lora`` and requests
    must not carry a ``lora_path``.
    """
    lora_enabled = lora_rank_of(model_config) > 0
    return lora_enabled and not model_config.lora.get("merge", False)


def sglang_lora_target_modules(target_modules: Any) -> list[str]:
    """Render verl's ``model.target_modules`` as SGLang's ``lora_target_modules``."""
    if target_modules == "all-linear":
        return ["all"]
    if isinstance(target_modules, str):
        raise ValueError(
            f"SGLang cannot serve a regex `target_modules` ({target_modules!r}); PEFT matches it "
            f"against the whole parameter key. Use `all-linear`, or list the module names."
        )
    return list(target_modules)


def broadcast_pyobj(
    data: list[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    """from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(get_device_name() if not force_cpu_device else "cpu")

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, dtype=np.uint8)).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


def _compact_for_bucket(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor safe to retain in a weight-sync bucket without pinning extra memory.

    ``get_named_tensor_buckets`` keeps every tensor alive until its bucket is flushed. A tensor
    that is a *view* into a larger backing buffer would therefore keep that whole buffer resident
    (and ship the whole buffer downstream), so such views must be compacted with ``clone()``.

    However the weights synced here come from ``DTensor.full_tensor()`` (a fresh all-gather) and
    already own tight, contiguous storage. Cloning those allocates a second full-size buffer and
    transiently doubles the tensor's footprint -- which OOMs on multi-GiB fused MoE weights
    (e.g. ``[num_experts, ...]`` ``gate_up_proj``/``qkv``) while the actor params and rollout
    weights are both already resident. Skip the clone when the tensor already owns its storage.
    """
    return tensor.clone() if tensor._base is not None else tensor


async def get_named_tensor_buckets(
    iterable: Iterator[tuple[str, torch.Tensor]], bucket_bytes: int
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """
    Group tensors into buckets based on a specified size in megabytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]

    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    from verl.utils.fusion_groups import fusion_key, group_size

    current_bucket = []
    current_size = 0
    # Members of a fused destination param (SGLang's DSv4 wqkv_a / wkv_gate) are
    # held here until the group is whole, then placed in one bucket: that loader
    # cats both halves inside a single load_weights call and asserts its cache
    # empty on return, so a bucket boundary between them makes it fire. A full
    # sync contains every member by construction, so this only ever delays a
    # tensor by its sibling -- nothing is dropped.
    staged: dict = {}

    def _place(entries):
        """Put an indivisible run of entries in a bucket, cutting before it if
        it would not fit rather than partway through."""
        nonlocal current_bucket, current_size
        total = sum(sz for _, _, sz in entries)
        if current_bucket and current_size + total > bucket_bytes:
            out, current_bucket, current_size = current_bucket, [], 0
            return out, entries
        return None, entries

    async for name, tensor in ensure_async_iterator(iterable):
        tensor_size = tensor.element_size() * tensor.numel()
        key = fusion_key(name)
        if key is not None:
            slot = staged.setdefault(key, [])
            slot.append((name, _compact_for_bucket(tensor), tensor_size))
            if len(slot) < group_size(key[1]):
                continue
            entries = staged.pop(key)
        else:
            entries = [(name, _compact_for_bucket(tensor), tensor_size)]

        flushed, entries = _place(entries)
        if flushed:
            yield flushed
        for n, t, sz in entries:
            current_bucket.append((n, t))
            current_size += sz

    assert not staged, (
        f"fused param groups never completed in this sync: {sorted(staged)}. "
        f"Every member must appear in the weight stream."
    )
    if current_bucket:
        yield current_bucket
