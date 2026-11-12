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

import json
import os
import shutil
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields
from itertools import islice
from pathlib import Path
from typing import Generator, Iterable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import convert_weight_keys

from .fsdp_model_merger import FSDPModelMerger, merge_non_dtensor_shards

_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
}
_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".index.json")
_PLAIN_FLOAT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class _ShardLocation:
    """Where one rank's local shard of a DTensor sits in the full tensor.

    VeOmni shards dense params on a 1-D FSDP mesh, and routed experts on a 1-D ``ep_fsdp`` mesh whose root
    mesh also carries the ``ep`` dim: each ep rank owns a contiguous run of experts along dim 0.
    """

    shard_rank: int
    ep_rank: int
    ep_size: int
    replica: bool


def _locate_shard(tensor: DTensor) -> _ShardLocation:
    mesh = tensor.device_mesh
    if mesh.ndim != 1 or len(tensor.placements) != 1:
        raise NotImplementedError(
            f"Only 1-D DTensor meshes are supported, got {mesh.mesh_dim_names} with {tensor.placements}"
        )
    root = mesh._get_root_mesh()
    root_coordinate = dict(zip(root.mesh_dim_names, root.get_coordinate(), strict=True))
    replica = any(coord != 0 for dim, coord in root_coordinate.items() if dim.endswith("replicate"))
    if "ep" in root_coordinate and mesh.mesh_dim_names[0] != "ep":
        ep_rank, ep_size = root_coordinate["ep"], root.size(root.mesh_dim_names.index("ep"))
    else:
        ep_rank, ep_size = 0, 1
    return _ShardLocation(mesh.get_coordinate()[0], ep_rank, ep_size, replica)


class _StreamedStateDict(Mapping):
    """A read-once state dict whose ``items()`` streams the merged tensors instead of holding them all."""

    def __init__(self, keys: list[str], items: Iterable[tuple[str, torch.Tensor]]):
        self._keys = keys
        self._items = items

    def __getitem__(self, key):
        raise KeyError(f"{key}: a streamed state dict can only be read through items()")

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def items(self):
        return self._items


class _SafetensorsShardWriter:
    """Write tensors into ``model-XXXXX-of-YYYYY.safetensors`` shards plus the index file.

    Each shard is saved in the background so disk writes overlap with merging; at most one save is in flight,
    which bounds host memory to about two shards.
    """

    def __init__(self, target_dir: str, max_shard_bytes: int = 5 * 1024**3):
        self.target_dir = target_dir
        self.max_shard_bytes = max_shard_bytes
        self.shard: dict[str, torch.Tensor] = {}
        self.shard_bytes = 0
        self.total_size = 0
        self.tmp_shards: list[tuple[str, list[str]]] = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: Optional[Future] = None

    def add(self, name: str, tensor: torch.Tensor):
        nbytes = tensor.numel() * tensor.element_size()
        if self.shard and self.shard_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self.shard[name] = tensor.detach().contiguous().cpu()
        self.shard_bytes += nbytes
        self.total_size += nbytes

    @staticmethod
    def _save(shard: dict[str, torch.Tensor], path: str):
        # Serialize in memory and write sequentially: save_file seeks, which FUSE mounts like HDFS reject.
        data = save(shard, metadata={"format": "pt"})
        with open(path, "wb") as f:
            f.write(data)

    def _flush(self):
        if self.pending is not None:
            self.pending.result()
        tmp_path = os.path.join(self.target_dir, f"model-{len(self.tmp_shards) + 1:05d}.safetensors.tmp")
        self.tmp_shards.append((tmp_path, list(self.shard)))
        self.pending = self.executor.submit(self._save, self.shard, tmp_path)
        self.shard, self.shard_bytes = {}, 0

    def close(self) -> dict[str, str]:
        if self.shard:
            self._flush()
        if self.pending is not None:
            self.pending.result()
        self.executor.shutdown()

        weight_map = {}
        for i, (tmp_path, names) in enumerate(self.tmp_shards, start=1):
            file_name = f"model-{i:05d}-of-{len(self.tmp_shards):05d}.safetensors"
            os.replace(tmp_path, os.path.join(self.target_dir, file_name))
            weight_map.update(dict.fromkeys(names, file_name))
        with open(os.path.join(self.target_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": self.total_size}, "weight_map": weight_map}, f, indent=2)
        return weight_map


class VeOmniModelMerger(FSDPModelMerger):
    """
    Model merger for checkpoints saved by the VeOmni engine.

    VeOmni saves FSDP2 DTensor shards like the FSDP engine, but with expert parallelism the fused routed-expert
    params (``mlp.experts.gate_up_proj`` / ``down_proj``) live on a separate ``ep_fsdp`` mesh, where each ep rank
    holds only its own experts. This merger reassembles both layouts from the per-rank files, streaming one tensor
    at a time from memory-mapped shards so models larger than host memory can be merged.

    The full weights are then exported exactly like ``VeOmniEngine.get_per_tensor_param`` exports them to the
    rollout engine and the ``hf_model`` checkpoint: through the model's VeOmni checkpoint tensor converter when it
    defines ``export_weights`` (e.g. DeepSeek-V4 re-quantizes into its original FP8/FP4 layout), and otherwise by
    restoring HF key names and splitting fused experts per expert.

    ``base_model_path`` is the model the training started from. Converters that export into the base checkpoint's
    layout need its index; when given, its non-weight files are copied, tensors the export does not produce (e.g.
    untrained MTP layers) are carried over from it, and every exported tensor is checked against it.

    Example:
        ```sh
        python -m verl.model_merger merge \
            --backend veomni \
            --local_dir checkpoints/<project>/<experiment>/global_step_10/actor \
            --base_model_path /path/to/DeepSeek-V4-Flash \
            --target_dir /path/to/merged_hf_model
        ```
    """

    def __init__(self, config, io_workers: int = 64, prefetch_tensors: int = 4):
        super().__init__(config)
        # Shard reads are latency-bound on network filesystems, so the pool is sized past the CPU count.
        self.io_executor = ThreadPoolExecutor(max_workers=io_workers)
        self.prefetch_tensors = prefetch_tensors

    def _load_rank_state_dicts(self, world_size: int) -> list[dict]:
        def load(rank: int) -> dict:
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            return torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)

        futures = [self.io_executor.submit(load, rank) for rank in range(world_size)]
        return [future.result() for future in tqdm(futures, desc=f"Opening {world_size} VeOmni shards")]

    def _merge_dtensor(self, key: str, shards: list[DTensor]) -> torch.Tensor:
        placement = shards[0].placements[0]
        if placement.is_partial():
            raise NotImplementedError(f"Partial placement is not supported: {key}")

        entries: dict[tuple[int, int], torch.Tensor] = {}
        ep_size = None
        for shard in shards:
            location = _locate_shard(shard)
            if ep_size is None:
                ep_size = location.ep_size
            assert location.ep_size == ep_size, f"{key}: inconsistent ep size across ranks"
            if location.replica or (placement.is_replicate() and location.shard_rank != 0):
                continue
            slot = (location.ep_rank, location.shard_rank)
            assert slot not in entries, f"{key}: ep rank {slot[0]} shard {slot[1]} is held by two ranks"
            entries[slot] = shard._local_tensor

        local_shape = list(shards[0].shape)
        full_shape = [local_shape[0] * ep_size, *local_shape[1:]]
        merged = torch.empty(full_shape, dtype=shards[0].dtype)

        copies = []
        for ep_rank in range(ep_size):
            ep_block = merged.narrow(0, ep_rank * local_shape[0], local_shape[0])
            ranks = sorted(shard_rank for er, shard_rank in entries if er == ep_rank)
            if placement.is_replicate():
                assert ranks == [0], f"{key}: missing replica for ep rank {ep_rank}"
                copies.append((ep_block, entries[(ep_rank, 0)]))
                continue
            assert ranks == list(range(len(ranks))), f"{key}: ep rank {ep_rank} is missing shards, got {ranks}"
            offset = 0
            for shard_rank in ranks:
                local = entries[(ep_rank, shard_rank)]
                copies.append((ep_block.narrow(placement.dim, offset, local.shape[placement.dim]), local))
                offset += local.shape[placement.dim]
            assert offset == local_shape[placement.dim], (
                f"{key}: shards of ep rank {ep_rank} cover {offset} of {local_shape[placement.dim]} rows"
            )

        for future in [self.io_executor.submit(dst.copy_, src) for dst, src in copies]:
            future.result()
        return merged

    def _merge_key(self, state_dicts: list[dict], key: str) -> torch.Tensor:
        shards = [state_dict[key] for state_dict in state_dicts]
        if isinstance(shards[0], DTensor):
            return self._merge_dtensor(key, shards)
        return merge_non_dtensor_shards(key, shards)

    def _iter_merged_tensors(
        self, state_dicts: list[dict], module_keys: dict[str, str]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield full tensors under their module key, reading the next few while the caller handles the current."""
        keys = iter(tqdm(list(module_keys), desc="Merging tensors"))
        with ThreadPoolExecutor(max_workers=self.prefetch_tensors) as prefetcher:
            pending = deque(
                (key, prefetcher.submit(self._merge_key, state_dicts, key))
                for key in islice(keys, self.prefetch_tensors)
            )
            while pending:
                key, future = pending.popleft()
                tensor = future.result()
                next_key = next(keys, None)
                if next_key is not None:
                    pending.append((next_key, prefetcher.submit(self._merge_key, state_dicts, next_key)))
                yield module_keys[key], tensor

    def _build_meta_module(self) -> torch.nn.Module:
        """Build the VeOmni model on the meta device, giving the converter and key names the engine uses."""
        from veomni.arguments import OpsImplementationConfig
        from veomni.models.auto import build_foundation_model
        from veomni.models.checkpoint_tensor_loading import prepare_fqn_to_index_mapping_for_model

        from verl.workers.engine.veomni.utils import load_safetensors_index

        # Nothing runs forward, so every op takes its reference implementation, which every model accepts.
        ops_implementation = OpsImplementationConfig(
            **{f.name: "eager" for f in fields(OpsImplementationConfig) if f.name.endswith("_implementation")}
        )
        module = build_foundation_model(
            config_path=self.hf_model_config_path,
            torch_dtype="float32",
            init_device="meta",
            ops_implementation=ops_implementation,
        )
        if self.config.base_model_path:
            prepare_fqn_to_index_mapping_for_model(module, load_safetensors_index(self.config.base_model_path))
        return module

    def _map_to_module_keys(self, module: torch.nn.Module, state_dict: dict) -> dict[str, str]:
        """Map checkpoint keys to the module's, renaming keys saved by older VeOmni versions like its loader does."""
        from veomni.models.checkpoint_tensor_loading import (
            get_checkpoint_tensor_converter,
            maybe_convert_checkpoint_tensor,
        )

        module_keys = set(module.state_dict())
        converter = get_checkpoint_tensor_converter(module)
        mapping = {}
        for key, value in state_dict.items():
            if key not in module_keys and converter is not None:
                converted = maybe_convert_checkpoint_tensor(
                    key, torch.empty(0, dtype=value.dtype, device="meta"), converter
                )
                mapping[key] = converted.name if converted is not None else key
            else:
                mapping[key] = key

        unmatched = set(mapping.values()) ^ module_keys
        if unmatched:
            raise ValueError(f"Checkpoint keys do not match the VeOmni model: {sorted(unmatched)[:10]}")
        return mapping

    def _iter_hf_weights(
        self, module: torch.nn.Module, tensors: Iterable[tuple[str, torch.Tensor]]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Same mapping as ``VeOmniEngine.get_per_tensor_param`` for models without a converter export."""
        from verl.workers.engine.veomni.utils import get_moe_param_handler

        names = {name: name for name in module.state_dict()}
        hf_names = {name: hf_name for hf_name, name in convert_weight_keys(names, module).items()}
        process_func = get_moe_param_handler(getattr(module.config, "model_type", "default"), ep_enabled=False)
        for name, tensor in tensors:
            name = hf_names[name]
            if "mlp.experts." in name:
                yield from process_func(name, tensor, expert_id_base=0)
            else:
                yield name, tensor

    def merge_and_save(self):
        if self.config.operation != "merge":
            raise NotImplementedError(f"VeOmni merger does not support operation {self.config.operation!r}")
        from veomni.models.checkpoint_tensor_loading import get_checkpoint_tensor_converter

        world_size = self._get_world_size()
        state_dicts = self._load_rank_state_dicts(world_size)
        module = self._build_meta_module()
        module_keys = self._map_to_module_keys(module, state_dicts[0])

        tensors = self._iter_merged_tensors(state_dicts, module_keys)

        # Same export dtype policy as the engine's hf_model checkpoint (FSDPCheckpointManager): a converter export
        # picks its own dtypes, otherwise fp32 master weights are narrowed to the bf16 compute dtype.
        hf_exporter = FSDPCheckpointManager.__new__(FSDPCheckpointManager)  # only its stateless name helpers are used
        tied_aliases = hf_exporter._get_tied_alias_names(module)
        converter = get_checkpoint_tensor_converter(module)
        if converter is not None and hasattr(converter, "export_weights"):
            if not self.config.base_model_path:
                raise ValueError(
                    f"{type(converter).__name__} exports weights in the base checkpoint layout; pass --base_model_path"
                )
            # Converter exports may quantize with accelerator kernels.
            device = torch.device(get_device_name(), get_device_id())
            tensors = ((name, tensor.to(device)) for name, tensor in tensors)
            # export_weights reads the module only through state_dict(), so point it at the merged tensors.
            module.state_dict = lambda *args, **kwargs: _StreamedStateDict(list(module_keys.values()), tensors)
            weights, export_dtype, preserved_names = converter.export_weights(module), None, set()
        else:
            weights, export_dtype = self._iter_hf_weights(module, tensors), torch.bfloat16
            preserved_names = hf_exporter._get_dtype_preserved_names(module)

        base_meta = self._read_base_meta() if self.config.base_model_path else {}
        writer = _SafetensorsShardWriter(self.config.target_dir)
        exported = set()
        for name, tensor in weights:
            if name in tied_aliases:
                continue
            if export_dtype is not None and tensor.dtype == torch.float32 and name not in preserved_names:
                tensor = tensor.to(export_dtype)
            if self.config.base_model_path:
                assert name in base_meta, f"{name}: exported weight is not in the base checkpoint"
                tensor = self._match_base_tensor(name, tensor, *base_meta[name])
            assert name not in exported, f"{name} exported twice"
            exported.add(name)
            writer.add(name, tensor)
        print(f"Exported {len(exported)} tensors")
        if self.config.base_model_path:
            self._carry_over_base_tensors(writer, exported | tied_aliases)
        writer.close()

        self._copy_non_weight_files()
        if self.config.hf_upload:
            self.upload_to_huggingface()

    @staticmethod
    def _match_base_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
        """Store the tensor like the base checkpoint does; only plain float precision may differ."""
        assert list(tensor.shape) == shape, f"{name}: exported shape {list(tensor.shape)}, base checkpoint has {shape}"
        if tensor.dtype != dtype:
            assert tensor.dtype in _PLAIN_FLOAT_DTYPES and dtype in _PLAIN_FLOAT_DTYPES, (
                f"{name}: exported {tensor.dtype}, base checkpoint has {dtype}"
            )
            tensor = tensor.to(dtype)
        return tensor

    def _base_weight_map(self) -> dict[str, str]:
        with open(os.path.join(self.config.base_model_path, "model.safetensors.index.json")) as f:
            return json.load(f)["weight_map"]

    def _read_base_meta(self) -> dict[str, tuple[torch.dtype, list[int]]]:
        def read(file_name: str) -> dict[str, tuple[torch.dtype, list[int]]]:
            with safe_open(os.path.join(self.config.base_model_path, file_name), framework="pt") as f:
                return {
                    name: (_SAFETENSORS_DTYPES[f.get_slice(name).get_dtype()], f.get_slice(name).get_shape())
                    for name in f.keys()
                }

        meta = {}
        for file_meta in self.io_executor.map(read, sorted(set(self._base_weight_map().values()))):
            meta.update(file_meta)
        return meta

    def _carry_over_base_tensors(self, writer: _SafetensorsShardWriter, skip: set[str]):
        """Copy base checkpoint tensors the export does not produce, e.g. MTP layers the training never touches."""
        by_file = defaultdict(list)
        for name, file_name in self._base_weight_map().items():
            if name not in skip:
                by_file[file_name].append(name)
        carried = sorted(name for names in by_file.values() for name in names)
        if carried:
            prefixes = Counter(name.split(".")[0] for name in carried)
            print(f"Carrying {len(carried)} untrained tensors over from the base model, by prefix: {dict(prefixes)}")
        for file_name, names in tqdm(sorted(by_file.items()), desc="Carrying over base model tensors"):
            with safe_open(os.path.join(self.config.base_model_path, file_name), framework="pt") as f:
                for name in names:
                    writer.add(name, f.get_tensor(name))

    def _copy_non_weight_files(self):
        src_dir = self.config.base_model_path or self.hf_model_config_path
        for entry in os.listdir(src_dir):
            src = os.path.join(src_dir, entry)
            if entry.startswith(".") or entry.endswith(_WEIGHT_FILE_SUFFIXES):
                continue
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(self.config.target_dir, entry), dirs_exist_ok=True)
            else:
                shutil.copy(src, self.config.target_dir)

    def cleanup(self):
        self.io_executor.shutdown()
