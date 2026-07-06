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
    if tensor.is_contiguous() and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size():
        return tensor
    return tensor.clone()


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

    current_bucket = []
    current_size = 0
    async for name, tensor in ensure_async_iterator(iterable):
        tensor_size = tensor.element_size() * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield current_bucket
            current_bucket = [(name, _compact_for_bucket(tensor))]
            current_size = tensor_size
        else:
            current_bucket.append((name, _compact_for_bucket(tensor)))
            current_size += tensor_size

    if current_bucket:
        yield current_bucket
