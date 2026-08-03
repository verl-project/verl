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

import asyncio
import logging
import re
import time
from typing import Any, Generator

import ray
import torch
from torch.distributed.tensor import DTensor

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
)

from .tpu_weight_registry import TPUWeightRegistry

logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION / CONSTANTS FOR WEIGHT TRANSFER ---
SYNC_LAYER_BY_LAYER = False  # Qwen3-0.6B is small, whole-model sync is fast and safe
TPU_COPY_CHUNK_SIZE_PARAMETERS = 30

# =====================================================================
# Namespace & Formatting Utilities
# =====================================================================


def get_clean_name(name: str) -> str:
    """Strip FSDP/DCP wrapper prefixes from state dict keys to match standard model namespaces."""
    return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("module.", "")


def get_layer_group(key: str) -> str:
    """Given a state dict key, returns its group name (e.g. 'embeddings', 'layers.0', 'output')."""
    clean_k = get_clean_name(key)
    match = re.search(r"layers\.(\d+)\.", clean_k)
    if match:
        return f"layers.{match.group(1)}"
    elif "tok_embeddings" in clean_k:
        return "embeddings"
    else:
        return "output"


# =====================================================================
# TPU Worker Weight Injection & Slicing
# =====================================================================


def load_weights_on_worker(vllm_model, state_dict: dict, rank: int) -> int:
    """
    Worker-side weight loader. Performs host-side CPU sharding (slicing)
    and chunked, memory-safe, JIT-partitioned PCIe copying to TPU.
    """
    t_start = time.perf_counter()

    if isinstance(state_dict, dict) and "grouped" in state_dict:
        grouped_dict = state_dict["grouped"]
    else:
        grouped_dict = {"all": state_dict}

    total_keys = 0
    from concurrent.futures import ThreadPoolExecutor

    temp_tpu_tensors = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for group_name, group_sd in grouped_dict.items():
            keys_loaded = _load_single_group_on_worker(
                vllm_model, group_sd, rank, executor=executor, temp_tpu_tensors=temp_tpu_tensors
            )
            total_keys += keys_loaded

    import torch_tpu

    torch_tpu._internal.sync.synchronize(wait=True)
    del temp_tpu_tensors
    import gc

    gc.collect()

    t_total = time.perf_counter() - t_start
    if rank == 0:
        logger.info(f"Worker 0: Loaded {total_keys} keys in {t_total:.3f}s")
    return total_keys


def _load_single_group_on_worker(vllm_model, group_sd: dict, rank: int, executor=None, temp_tpu_tensors=None) -> int:
    flat_tensors = group_sd["flat_tensors"]
    metadata = group_sd["metadata"]

    clean_metadata = {}
    num_keys = 0
    for dtype, items in metadata.items():
        clean_items = []
        offset = 0
        for k, shape, numel in items:
            clean_k = get_clean_name(k)
            clean_items.append((clean_k, shape, numel, offset))
            num_keys += 1

            if "tok_embeddings.weight" in clean_k:
                lm_k = clean_k.replace("tok_embeddings", "lm_head")
                clean_items.append((lm_k, shape, numel, offset))
                num_keys += 1

            offset += numel
        clean_metadata[dtype] = clean_items

    model_sd = vllm_model.model.state_dict()

    def resolve_key(k):
        if k in model_sd:
            return k
        if k.startswith("model.") and k[6:] in model_sd:
            return k[6:]
        if f"model.{k}" in model_sd:
            return f"model.{k}"
        return k

    for dtype, flat_data in flat_tensors.items():
        items = clean_metadata.get(dtype, [])
        if not items:
            continue

        flat_cpu = torch.from_numpy(flat_data) if not isinstance(flat_data, torch.Tensor) else flat_data
        if dtype == torch.bfloat16 and flat_cpu.dtype == torch.int16:
            # COMMENT: On TPU CPU builds (e.g. torch_tpu), converting torch.bfloat16 to numpy raises a TypeError.
            # Thus, we serialize it as int16 (same bit representation) and view it back to bfloat16 here.
            # TODO: remove HACK once PyTorch CPU native bfloat16 to numpy conversion is universally stable.
            flat_cpu = flat_cpu.view(torch.bfloat16)

        local_items = []
        local_tensors_to_cat = []
        local_offset = 0

        def process_item_parallel(item, flat_cpu=flat_cpu):
            k, shape, numel, offset = item
            target_key = resolve_key(k)
            if target_key not in model_sd:
                return None

            target_v = model_sd[target_key]
            target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v

            param_cpu_global = flat_cpu[offset : offset + numel].view(shape)
            if target_local.shape == shape:
                param_cpu_local = param_cpu_global
            else:
                sharded = False
                for dim in range(len(shape)):
                    if shape[dim] != target_local.shape[dim]:
                        shard_size = target_local.shape[dim]
                        rank_offset = shard_size * rank
                        indices = [slice(None)] * len(shape)
                        indices[dim] = slice(rank_offset, rank_offset + shard_size)
                        # COMMENT: Removed .clone() to enable zero-copy views during slicing.
                        param_cpu_local = param_cpu_global[tuple(indices)]
                        sharded = True
                        break
                if not sharded:
                    param_cpu_local = param_cpu_global

            return (target_key, target_local.shape, target_local.numel(), param_cpu_local.reshape(-1))

        if executor is None:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=8) as local_exec:
                sliced_results = list(local_exec.map(process_item_parallel, items))
        else:
            sliced_results = list(executor.map(process_item_parallel, items))

        for res in sliced_results:
            if res is None:
                continue
            target_key, target_shape, target_numel, param_cpu_local_flat = res
            local_tensors_to_cat.append(param_cpu_local_flat)
            local_items.append((target_key, target_shape, target_numel, local_offset))
            local_offset += target_numel

        if not local_tensors_to_cat:
            continue

        flat_local_cpu = torch.cat(local_tensors_to_cat)

        chunks = [
            local_items[i : i + TPU_COPY_CHUNK_SIZE_PARAMETERS]
            for i in range(0, len(local_items), TPU_COPY_CHUNK_SIZE_PARAMETERS)
        ]

        for chunk in chunks:
            chunk_start_offset = chunk[0][3]
            chunk_end_offset = chunk[-1][3] + chunk[-1][2]
            flat_chunk_cpu = flat_local_cpu[chunk_start_offset:chunk_end_offset]

            flat_chunk_tpu = flat_chunk_cpu.to("tpu")
            if temp_tpu_tensors is not None:
                temp_tpu_tensors.append(flat_chunk_tpu)

            for target_key, local_shape, local_numel, offset in chunk:
                local_offset = offset - chunk_start_offset
                slice_tpu = flat_chunk_tpu[local_offset : local_offset + local_numel].view(local_shape)

                target_v = model_sd[target_key]
                target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
                target_local.copy_(slice_tpu)

            if temp_tpu_tensors is None:
                del flat_chunk_tpu
                import torch_tpu

                torch_tpu._internal.sync.synchronize(wait=True)

    return num_keys


# =====================================================================
# TPUCheckpointEngine Registration
# =====================================================================


@CheckpointEngineRegistry.register("tpu")
class TPUCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs) -> None:
        self.is_master = is_master
        self.bucket_size = bucket_size

        # Connect to or create named Ray TPUWeightRegistry actor
        try:
            self.registry = ray.get_actor("TPUWeightRegistry", namespace="verl")
        except ValueError:
            try:
                # COMMENT: Since TPUWeightRegistry is already a remote class decorated with @ray.remote,
                # wrapping it with ray.remote(TPUWeightRegistry) throws a TypeError.
                # Calling TPUWeightRegistry.options directly is the correct way to specify options.
                # TODO: remove HACK once a unified and clean TPU checkpoint/weight registry engine is standard.
                self.registry = TPUWeightRegistry.options(
                    name="TPUWeightRegistry", namespace="verl", lifetime="detached"
                ).remote()
            except Exception:
                self.registry = ray.get_actor("TPUWeightRegistry", namespace="verl")

    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        return {}, {}

    def init_process_group(self, **kwargs):
        pass

    def finalize(self):
        pass

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        t_start = time.perf_counter()

        try:
            import torch_tpu

            torch_tpu._internal.sync.synchronize(wait=True)
        except Exception:
            pass

        if not self.is_master:
            # Non-master ranks must consume the generator to prevent hangs
            for _ in weights:
                pass
            return

        step_key = global_steps if global_steps is not None else 0
        logger.info(f"@@@ TPUCheckpointEngine: [Step {step_key}] Start send_weights...")

        # Time generator consumption and CPU offloading
        t_offload_start = time.perf_counter()
        grouped_weights = {}
        for k, v in weights:
            cpu_v = v.detach().cpu()
            if "layers." in k:
                # Extract layer part: model.layers.12.self_attn... -> model.layers.12
                parts = k.split(".")
                idx = parts.index("layers")
                group_name = ".".join(parts[: idx + 2])
            else:
                group_name = "other"
            grouped_weights.setdefault(group_name, []).append((k, cpu_v))
        t_offload = time.perf_counter() - t_offload_start

        # Time grouping and flattening
        t_group_start = time.perf_counter()
        grouped_dict = {}
        for group_name, group_items in grouped_weights.items():
            by_dtype = {}
            for k, cpu_v in group_items:
                by_dtype.setdefault(cpu_v.dtype, []).append((k, cpu_v))

            flat_tensors = {}
            metadata = {}
            for dtype, items in by_dtype.items():
                flat_cpu = torch.cat([v.view(-1) for _, v in items])
                if dtype == torch.bfloat16:
                    flat_tensors[dtype] = flat_cpu.view(torch.int16).numpy()
                else:
                    flat_tensors[dtype] = flat_cpu.numpy()
                metadata[dtype] = [(k, v.shape, v.numel()) for k, v in items]

            grouped_dict[group_name] = {"flat_tensors": flat_tensors, "metadata": metadata}

        state_dict = {"grouped": grouped_dict}
        t_group = time.perf_counter() - t_group_start

        # Time Ray Put upload
        t_put_start = time.perf_counter()
        ref = ray.put(state_dict)
        t_put = time.perf_counter() - t_put_start

        # Time Registry update
        t_reg_start = time.perf_counter()
        await self.registry.set_weights.remote(step_key, ref)
        t_reg = time.perf_counter() - t_reg_start

        t_total = time.perf_counter() - t_start
        logger.debug(
            f"TPUCheckpointEngine Phase A [Step {step_key}]: Total={t_total:.3f}s, "
            f"Offload={t_offload:.3f}s, GroupFlatten={t_group:.3f}s, RayPut={t_put:.3f}s, Registry={t_reg:.3f}s"
        )

    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # Rollout uses load_weights_from_ray_registry directly, receive_weights is unused
        raise NotImplementedError("Rollout on TPU uses direct load_weights_from_ray_registry via collective_rpc.")


async def update_tpu_weights(manager, global_steps: int | None = None) -> dict:
    """Synchronize weights from actor worker group to rollout replicas on TPU."""
    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning(f"Failed to abort replicas at step {global_steps}: {e}")
    t_abort = time.perf_counter() - t_abort_start

    t_total_start = time.perf_counter()
    # 1. Extract and upload weights on trainer side (Rank 0 sends to Ray Plasma)
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode=manager.backend)
    if isinstance(actor_refs, list):
        await asyncio.gather(*actor_refs)
    elif actor_refs is not None:
        await actor_refs

    # 2. Call collective_rpc on all rollout replicas to load weights from the registry
    step_key = global_steps if global_steps is not None else 0
    futures = [
        replica.server_handle.collective_rpc.remote(method="load_weights_from_ray_registry", args=(step_key,))
        for replica in manager.replicas
    ]
    await asyncio.gather(*futures)
    t_total = time.perf_counter() - t_total_start

    logger.info(f"TPU weight sync for step {global_steps} completed in {t_total + t_abort:.3f}s")

    await manager.resume_generation_replicas()
    return {}
