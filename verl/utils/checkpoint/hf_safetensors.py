# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import json
import os
from collections.abc import Collection, Iterable

import torch
from safetensors.torch import save_file

HF_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def export_hf_safetensors(
    weights: Iterable[tuple[str, torch.Tensor]],
    output_dir: str,
    max_shard_bytes: int,
    rank: int = 0,
    num_writers: int = 1,
    exclude_names: Collection[str] = (),
    target_dtype: torch.dtype | None = None,
    preserve_dtype_names: Collection[str] = (),
) -> tuple[int, int, int]:
    """Stream named tensors into HuggingFace safetensors shards.

    This function is used by ``FSDPCheckpointManager`` when saving an HF model and
    ``FileSystemCheckpointEngine`` when exporting weights for rollout.

    To improve ``FileSystemCheckpointEngine`` export performance, this function supports distributed write, where
    multiple ranks write different shards of the same HF model in parallel. ``FSDPCheckpointManager`` does not use
    distributed write and still writes all shards on rank 0.

    Shard ownership is ``shard_index % num_writers == rank``:

    - Without distributed write, ``num_writers=1``, so rank 0 owns every shard.
    - With distributed write, ``num_writers`` is the actor world size and shards are assigned round-robin. For
      example, with four writers and eight shards, rank 0 writes shards 0 and 4, rank 1 writes 1 and 5, rank 2 writes
      2 and 6, and rank 3 writes 3 and 7.
    """
    shard, shard_names, shard_bytes, total_size = {}, [], 0, 0
    tmp_shards: list[tuple[str, list[str]]] = []

    def flush_shard() -> None:
        nonlocal shard, shard_names, shard_bytes
        shard_index = len(tmp_shards)
        tmp_path = os.path.join(output_dir, f"model-{shard_index + 1:05d}.safetensors.tmp")
        if shard_index % num_writers == rank:
            save_file(shard, tmp_path, metadata={"format": "pt"})
        tmp_shards.append((tmp_path, shard_names))
        shard, shard_names, shard_bytes = {}, [], 0

    for name, tensor in weights:
        if name in exclude_names:
            continue
        if target_dtype is not None and tensor.dtype == torch.float32 and name not in preserve_dtype_names:
            tensor = tensor.to(target_dtype)
        nbytes = tensor.numel() * tensor.element_size()
        if shard_names and shard_bytes + nbytes > max_shard_bytes:
            flush_shard()

        if len(tmp_shards) % num_writers == rank:
            # Yielded tensors may be views of a buffer the generator overwrites later (EP broadcast),
            # so copy them out before advancing.
            shard[name] = tensor.detach().contiguous().to("cpu", copy=True)
        shard_names.append(name)
        shard_bytes += nbytes
        total_size += nbytes

    if shard_names:
        flush_shard()

    num_shards = len(tmp_shards)
    weight_map = {}
    for shard_index, (tmp_path, names) in enumerate(tmp_shards):
        file_name = f"model-{shard_index + 1:05d}-of-{num_shards:05d}.safetensors"
        if shard_index % num_writers == rank:
            os.replace(tmp_path, os.path.join(output_dir, file_name))
        weight_map.update(dict.fromkeys(names, file_name))

    with open(os.path.join(output_dir, HF_WEIGHTS_INDEX_NAME), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    return total_size, num_shards, len(weight_map)
