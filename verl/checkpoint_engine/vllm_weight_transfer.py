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
"""CheckpointEngine integration for vLLM's native WeightTransferEngine backends."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping
from functools import cached_property
from typing import Any

import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.utils.net_utils import get_free_port

try:
    from vllm.distributed.weight_transfer.base import WeightSource as _WeightSourceBase
    from vllm.distributed.weight_transfer.base import _dtensor_layout as _vllm_dtensor_layout
except ImportError:
    _WeightSourceBase = object
    _vllm_dtensor_layout = None

logger = logging.getLogger(__name__)

# Mapping from verl checkpoint_engine.backend name to vLLM WeightTransferConfig.backend.
VLLM_NATIVE_BACKENDS: dict[str, str] = {
    "vllm_ipc": "ipc",
    "vllm_nccl": "nccl",
    "vllm_sparse_nccl": "sparse_nccl",
    "vllm_sharded_rdt": "sharded_rdt",
    "vllm_nccl_m2n": "nccl_m2n",
}

_SPARSE_MAX_ENTRY_ELEMS = 64 << 20


def resolve_vllm_weight_transfer_backend(
    checkpoint_backend: str | None,
    engine_kwargs: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve the vLLM weight_transfer_config backend from checkpoint_engine and engine_kwargs."""
    if checkpoint_backend == "delta_sharded":
        raise ValueError(
            "checkpoint_engine.backend='delta_sharded' is no longer used with vLLM; "
            "use checkpoint_engine.backend='vllm_sparse_nccl' instead."
        )

    configured = None
    if engine_kwargs is not None and "weight_transfer_config" in engine_kwargs:
        wtc = engine_kwargs["weight_transfer_config"]
        if wtc is not None:
            if not isinstance(wtc, Mapping):
                raise TypeError("rollout.engine_kwargs.vllm.weight_transfer_config must be a mapping")
            configured = wtc.get("backend")

    expected = VLLM_NATIVE_BACKENDS.get(checkpoint_backend or "")
    if expected is not None:
        if configured is not None and configured != expected:
            raise ValueError(
                f"checkpoint_engine.backend={checkpoint_backend!r} requires vLLM weight_transfer_config "
                f"backend={expected!r}, got {configured!r}"
            )
        return expected

    return configured


class VerlRayVLLMWeightSyncClient:
    """Structural VLLMWeightSyncClient fanning out to one or more vLLMHttpServer Ray actors."""

    def __init__(self, server_handles: list[Any], replica_world_sizes: list[int]) -> None:
        if not server_handles or len(server_handles) != len(replica_world_sizes):
            raise ValueError("server_handles and replica_world_sizes must be non-empty and have matching lengths")
        self.server_handles = list(server_handles)
        self.replica_world_sizes = [int(ws) for ws in replica_world_sizes]

    @property
    def total_workers(self) -> int:
        return sum(self.replica_world_sizes)

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        if "rank_offset" in init_info:
            offset = int(init_info["rank_offset"])
            refs = []
            for handle, ws in zip(self.server_handles, self.replica_world_sizes, strict=True):
                payload = {**init_info, "rank_offset": offset}
                refs.append(handle.init_weight_transfer_engine.remote(payload))
                offset += ws
            ray.get(refs)
            return

        if "num_consumers" in init_info and len(self.server_handles) > 1:
            if len(set(self.replica_world_sizes)) != 1:
                raise ValueError(
                    f"vllm_sharded_rdt requires uniform world sizes across replicas; got {self.replica_world_sizes}"
                )
            num_replicas = len(self.server_handles)
            refs = [
                handle.init_weight_transfer_engine.remote(
                    {**init_info, "replica_rank": idx, "num_replicas": num_replicas}
                )
                for idx, handle in enumerate(self.server_handles)
            ]
            ray.get(refs)
            return

        ray.get([handle.init_weight_transfer_engine.remote(init_info) for handle in self.server_handles])

    def shutdown_weight_transfer_engine(self) -> None:
        ray.get([handle.shutdown_weight_transfer_engine.remote() for handle in self.server_handles])

    def start_weight_update(self) -> None:
        ray.get([handle.start_weight_update.remote() for handle in self.server_handles])

    def update_weights(self, update_info: dict[str, Any] | list[dict[str, Any]]) -> None:
        ray.get([handle.update_weights.remote(update_info) for handle in self.server_handles])

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        ray.get([handle.finish_weight_update.remote(weight_version) for handle in self.server_handles])


def _require_no_unmerged_lora(peft_config: Any) -> None:
    if peft_config is not None:
        raise NotImplementedError(
            "vLLM native weight transfer does not support unmerged LoRA adapters; "
            "set actor_rollout_ref.model.lora.merge=True or checkpoint_engine.backend='naive'."
        )


def _iter_engine_params(engine: Any, layered_summon: bool = False) -> Iterator[tuple[str, torch.Tensor]]:
    per_tensor_param, peft_config = engine.get_per_tensor_param(layered_summon=layered_summon, base_sync_done=True)
    _require_no_unmerged_lora(peft_config)
    return iter(per_tensor_param)


class VerlEngineWeightSource(_WeightSourceBase):
    """WeightSource adapter over a verl training engine's HF parameter export."""

    def __init__(
        self,
        engine: Any,
        *,
        layered_summon: bool = False,
        use_local_shards: bool = False,
        trainer_world_size: int = 1,
    ) -> None:
        self._engine = engine
        self._layered_summon = layered_summon
        self._use_local_shards = use_local_shards
        self._trainer_world_size = trainer_world_size

    @cached_property
    def metadata(self) -> list[Any]:
        from vllm.distributed.weight_transfer.base import ParamMeta

        metas = []
        if self._use_local_shards:
            shards, peft_config = self._engine.get_per_tensor_param_shard()
            _require_no_unmerged_lora(peft_config)
            all_ranks = tuple(range(self._trainer_world_size))
            for name, param in shards:
                layout = _vllm_dtensor_layout(param) if _vllm_dtensor_layout is not None else None
                if layout is None:
                    metas.append(ParamMeta(name, param.dtype, tuple(param.shape), owners=all_ranks))
                else:
                    owners, mesh_shape, placements = layout
                    metas.append(
                        ParamMeta(
                            name,
                            param.dtype,
                            tuple(param.shape),
                            owners=owners,
                            mesh_shape=mesh_shape,
                            placements=placements,
                        )
                    )
            return metas

        for name, tensor in _iter_engine_params(self._engine, layered_summon=self._layered_summon):
            metas.append(ParamMeta(name, tensor.dtype, tuple(tensor.shape)))
        return metas

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        return _iter_engine_params(self._engine, layered_summon=self._layered_summon)

    def iter_local(self) -> Iterator[tuple[str, torch.Tensor]]:
        shards, peft_config = self._engine.get_per_tensor_param_shard()
        _require_no_unmerged_lora(peft_config)
        for name, param in shards:
            layout = _vllm_dtensor_layout(param) if _vllm_dtensor_layout is not None else None
            if layout is None:
                yield name, param
            elif layout[2] is None:
                yield name, param.full_tensor()
            else:
                yield name, param.to_local()


def _get_trainer_rank_and_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _broadcast_rendezvous_endpoint(rank: int) -> tuple[str, int]:
    """Allocate a TCPStore endpoint on rank 0 and broadcast it to all trainer ranks."""
    endpoint: list[Any] = [None]
    if rank == 0:
        ip = ray.util.get_node_ip_address().strip("[]")
        port, _ = get_free_port(ip)
        endpoint[0] = (ip, int(port))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(endpoint, src=0)
    return endpoint[0]


def _target_signature(server_handles: list[Any], replica_world_sizes: list[int]) -> tuple[tuple[Any, int], ...]:
    keys = []
    for handle, ws in zip(server_handles, replica_world_sizes, strict=True):
        actor_id = getattr(handle, "_ray_actor_id", None)
        keys.append((actor_id.hex() if hasattr(actor_id, "hex") else id(handle), int(ws)))
    return tuple(keys)


class VLLMWeightTransferCheckpointEngine(CheckpointEngine):
    """CheckpointEngine delegating transport and reload directly to vLLM's native WeightTransferEngine."""

    is_vllm_native = True
    requires_engine = True
    is_colocated = False
    vllm_backend: str = ""

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        rebuild_group: bool = False,
        layered_summon: bool = False,
        **trainer_kwargs: Any,
    ) -> None:
        self.bucket_size = int(bucket_size)
        self.is_master = is_master
        self.rebuild_group = rebuild_group
        self.layered_summon = layered_summon
        self.trainer_kwargs = dict(trainer_kwargs)

        self._client: VerlRayVLLMWeightSyncClient | None = None
        self._target_sig: tuple[tuple[Any, int], ...] | None = None
        self._trainer_engine: Any = None
        self._weight_source: VerlEngineWeightSource | None = None

    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        return {}, {}

    def init_process_group(self, **kwargs: Any) -> None:
        pass

    def configure_targets(self, server_handles: list[Any], replica_world_sizes: list[int]) -> None:
        """Bind target vLLM server actors; re-initializes if the replica set changed."""
        sig = _target_signature(server_handles, replica_world_sizes)
        if sig == self._target_sig and self._client is not None:
            return
        self._shutdown_trainer_engine()
        self._client = VerlRayVLLMWeightSyncClient(server_handles, replica_world_sizes)
        self._target_sig = sig

    def _shutdown_trainer_engine(self) -> None:
        if self._trainer_engine is None:
            return
        rank, _ = _get_trainer_rank_and_world_size()
        try:
            self._trainer_engine.shutdown()
        finally:
            self._trainer_engine = None
        if rank == 0 and self._client is not None:
            self._client.shutdown_weight_transfer_engine()

    def finalize(self) -> None:
        if self.rebuild_group:
            self._shutdown_trainer_engine()

    def _build_trainer_init_info(
        self,
        rank: int,
        trainer_world_size: int,
        client: VerlRayVLLMWeightSyncClient,
    ) -> Any:
        raise NotImplementedError

    def _ensure_trainer_engine(self, engine: Any) -> Any:
        if self._trainer_engine is not None:
            return self._trainer_engine
        if self._client is None:
            raise RuntimeError("configure_targets() must be called before send_weights()")

        from vllm.distributed.weight_transfer import WeightTransferTrainerFactory

        rank, trainer_world_size = _get_trainer_rank_and_world_size()
        source = None
        if self.vllm_backend != "sparse_nccl":
            if self._weight_source is None or self._weight_source._engine is not engine:
                self._weight_source = VerlEngineWeightSource(
                    engine,
                    layered_summon=self.layered_summon,
                    use_local_shards=(self.vllm_backend == "nccl_m2n"),
                    trainer_world_size=trainer_world_size,
                )
            source = self._weight_source

        init_info = self._build_trainer_init_info(rank, trainer_world_size, self._client)
        self._trainer_engine = WeightTransferTrainerFactory.trainer_init(
            init_info=init_info,
            client=self._client,
            source=source,
        )
        return self._trainer_engine

    async def send_weights(self, engine: Any, global_steps: int | None = None) -> dict[str, float] | None:
        t0 = time.perf_counter()
        trainer_engine = self._ensure_trainer_engine(engine)
        trainer_engine.send_weights()
        if self.is_master:
            return {"checkpoint_engine/sync_time_s": time.perf_counter() - t0}
        return None

    async def receive_weights(self, global_steps: int | None = None):
        raise NotImplementedError("vLLM native weight transfer delivers weights directly to vLLM workers")


@CheckpointEngineRegistry.register("vllm_ipc")
class VLLMIPCCheckpointEngine(VLLMWeightTransferCheckpointEngine):
    """Colocated CUDA IPC weight transfer via vLLM's native IPCWeightTransferEngine."""

    vllm_backend = "ipc"
    is_colocated = True

    def _build_trainer_init_info(
        self,
        rank: int,
        trainer_world_size: int,
        client: VerlRayVLLMWeightSyncClient,
    ) -> Any:
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

        return IPCTrainerInitInfo(
            rank=rank,
            packed=bool(self.trainer_kwargs.get("packed", True)),
            packed_buffer_size_bytes=int(self.trainer_kwargs.get("packed_buffer_size_bytes", self.bucket_size)),
        )


@CheckpointEngineRegistry.register("vllm_nccl")
class VLLMNCCLCheckpointEngine(VLLMWeightTransferCheckpointEngine):
    """Disaggregated dense NCCL weight transfer via vLLM's native NCCLWeightTransferEngine."""

    vllm_backend = "nccl"

    def _build_trainer_init_info(
        self,
        rank: int,
        trainer_world_size: int,
        client: VerlRayVLLMWeightSyncClient,
    ) -> Any:
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        master_address, master_port = _broadcast_rendezvous_endpoint(rank)
        return NCCLTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=1 + client.total_workers,
            rank=rank,
            packed=bool(self.trainer_kwargs.get("packed", True)),
            packed_buffer_size_bytes=int(self.trainer_kwargs.get("packed_buffer_size_bytes", self.bucket_size)),
            packed_num_buffers=int(self.trainer_kwargs.get("packed_num_buffers", 2)),
            use_load_plan=bool(self.trainer_kwargs.get("use_load_plan", False)),
        )


@CheckpointEngineRegistry.register("vllm_sharded_rdt")
class VLLMShardedRDTCheckpointEngine(VLLMWeightTransferCheckpointEngine):
    """Pull-based NIXL/RDT weight transfer via vLLM's native ShardedRDTWeightTransferEngine."""

    vllm_backend = "sharded_rdt"

    def _build_trainer_init_info(
        self,
        rank: int,
        trainer_world_size: int,
        client: VerlRayVLLMWeightSyncClient,
    ) -> Any:
        from vllm.distributed.weight_transfer.sharded_rdt_trainer import ShardedRDTTrainerInitInfo

        uniform = len(set(client.replica_world_sizes)) == 1
        default_wpr = client.replica_world_sizes[0] if uniform else 0
        namespace = self.trainer_kwargs.get("trainer_actor_namespace")
        if namespace is None and ray.is_initialized():
            namespace = ray.get_runtime_context().namespace
        kwargs = {
            k: v for k, v in self.trainer_kwargs.items() if k not in {"trainer_actor_namespace", "workers_per_replica"}
        }
        return ShardedRDTTrainerInitInfo(
            rank=rank,
            num_consumers=client.total_workers,
            workers_per_replica=int(self.trainer_kwargs.get("workers_per_replica", default_wpr)),
            trainer_actor_namespace=namespace,
            **kwargs,
        )


@CheckpointEngineRegistry.register("vllm_nccl_m2n")
class VLLMNCCLM2NCheckpointEngine(VLLMWeightTransferCheckpointEngine):
    """Disaggregated M2N resharding via vLLM's native M2NWeightTransferEngine."""

    vllm_backend = "nccl_m2n"

    def _build_trainer_init_info(
        self,
        rank: int,
        trainer_world_size: int,
        client: VerlRayVLLMWeightSyncClient,
    ) -> Any:
        from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo

        master_address, master_port = _broadcast_rendezvous_endpoint(rank)
        kwargs = dict(self.trainer_kwargs)
        if "dst_mesh_dims" in kwargs and kwargs["dst_mesh_dims"] is not None:
            kwargs["dst_mesh_dims"] = tuple(kwargs["dst_mesh_dims"])
        return M2NTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=trainer_world_size + client.total_workers,
            num_trainer_ranks=trainer_world_size,
            rank=rank,
            **kwargs,
        )


@CheckpointEngineRegistry.register("vllm_sparse_nccl")
class VLLMSparseNCCLCheckpointEngine(VLLMWeightTransferCheckpointEngine):
    """Disaggregated sparse delta transfer via vLLM's native SparseNCCLWeightTransferEngine."""

    vllm_backend = "sparse_nccl"

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        rebuild_group: bool = False,
        layered_summon: bool = False,
        rollout_dtype: torch.dtype | str = torch.bfloat16,
        batch_gather: int = 32,
        skip_seed: bool = False,
        **trainer_kwargs: Any,
    ) -> None:
        super().__init__(
            bucket_size=bucket_size,
            is_master=is_master,
            rebuild_group=rebuild_group,
            layered_summon=layered_summon,
            **trainer_kwargs,
        )
        self.rollout_dtype = getattr(torch, rollout_dtype) if isinstance(rollout_dtype, str) else rollout_dtype
        self.batch_gather = int(batch_gather)
        self.skip_seed = bool(skip_seed)
        self._shard_seeded = False

    def _build_trainer_init_info(
        self,
        rank: int,
        trainer_world_size: int,
        client: VerlRayVLLMWeightSyncClient,
    ) -> Any:
        from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseNCCLTrainerInitInfo

        master_address, master_port = _broadcast_rendezvous_endpoint(rank)
        return SparseNCCLTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=1 + client.total_workers,
            rank=rank,
        )

    def _send_patch_stream(
        self, trainer_engine: Any, pending: list[Any], pending_bytes: int, patch: Any
    ) -> tuple[int, bool]:
        patch_bytes = int(patch.indices.numel()) * (torch.int32.itemsize + patch.values.element_size())
        flushed = False
        if pending and pending_bytes + patch_bytes > self.bucket_size:
            trainer_engine.send_weight_chunk(pending)
            pending.clear()
            pending_bytes = 0
            flushed = True
        pending.append(patch)
        return pending_bytes + patch_bytes, flushed

    def _send_full_seed(
        self, trainer_engine: Any, weights: Iterator[tuple[str, torch.Tensor]]
    ) -> dict[str, float] | None:
        from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch

        rank, _ = _get_trainer_rank_and_world_size()
        is_sender = rank == 0
        pending: list[Any] = []
        pending_bytes = 0
        started = False
        flushes = 0
        total_elems = 0
        wire_bytes = 0

        for name, tensor in weights:
            tensor = tensor.detach()
            if not tensor.is_floating_point():
                continue
            if tensor.dtype != self.rollout_dtype:
                tensor = tensor.to(self.rollout_dtype)
            if not is_sender:
                del tensor
                continue
            flat = tensor.contiguous().reshape(-1)
            numel = int(flat.numel())
            total_elems += numel
            shape = tuple(tensor.shape)
            for start in range(0, numel, _SPARSE_MAX_ENTRY_ELEMS):
                end = min(start + _SPARSE_MAX_ENTRY_ELEMS, numel)
                idx = torch.arange(start, end, dtype=torch.int32, device=flat.device)
                vals = flat[start:end].contiguous()
                patch = SparseWeightPatch(name=name, indices=idx, values=vals, full_shape=shape)
                wire_bytes += (end - start) * (4 + vals.element_size())
                if not started:
                    assert self._client is not None
                    self._client.start_weight_update()
                    started = True
                pending_bytes, flushed = self._send_patch_stream(trainer_engine, pending, pending_bytes, patch)
                if flushed:
                    flushes += 1

        if not is_sender:
            return None
        if pending:
            trainer_engine.send_weight_chunk(pending)
            pending.clear()
            flushes += 1
        if started:
            assert self._client is not None
            self._client.finish_weight_update()
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": 1.0,
            "checkpoint_engine/changed_elems": float(total_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(flushes),
        }

    async def send_weights(self, engine: Any, global_steps: int | None = None) -> dict[str, float] | None:
        from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch

        from verl.checkpoint_engine.delta_checkpoint_engine import _GatherQueue

        trainer_engine = self._ensure_trainer_engine(engine)
        if not self._shard_seeded:
            metrics = None
            if not self.skip_seed:
                metrics = self._send_full_seed(
                    trainer_engine, _iter_engine_params(engine, layered_summon=self.layered_summon)
                )
            engine.prime_delta_snapshots()
            self._shard_seeded = True
            return metrics

        rank, _ = _get_trainer_rank_and_world_size()
        is_sender = rank == 0
        weights, _ = engine.get_per_tensor_param_delta_shard()
        pending: list[Any] = []
        pending_bytes = 0
        started = False
        flushes = 0
        changed_elems = 0
        total_elems = 0
        wire_bytes = 0

        def _consume(
            name: str,
            _dtype_str: str,
            full_shape: tuple[int, ...],
            full_numel: int,
            aidx: torch.Tensor | None,
            aval: torch.Tensor | None,
        ) -> None:
            nonlocal pending_bytes, started, flushes, changed_elems, total_elems, wire_bytes
            total_elems += int(full_numel)
            if aidx is None or aval is None or aidx.numel() == 0:
                return
            aidx = aidx.contiguous().to(dtype=torch.int32)
            aval = aval.contiguous()
            changed_elems += int(aidx.numel())
            wire_bytes += int(aidx.numel()) * (4 + aval.element_size())
            for start in range(0, int(aidx.numel()), _SPARSE_MAX_ENTRY_ELEMS):
                end = min(start + _SPARSE_MAX_ENTRY_ELEMS, int(aidx.numel()))
                patch = SparseWeightPatch(
                    name=name,
                    indices=aidx[start:end],
                    values=aval[start:end],
                    full_shape=tuple(full_shape),
                )
                if not started:
                    assert self._client is not None
                    self._client.start_weight_update()
                    started = True
                pending_bytes, flushed = self._send_patch_stream(trainer_engine, pending, pending_bytes, patch)
                if flushed:
                    flushes += 1

        gq = _GatherQueue(self.batch_gather, self.bucket_size, is_sender, _consume)
        for slots, dtype_str, counts, hf_idx, hf_val, pg in weights:
            gq.put(pg, slots, dtype_str, counts, hf_idx, hf_val)
        gq.flush_all()

        if not is_sender:
            return None
        if pending:
            trainer_engine.send_weight_chunk(pending)
            pending.clear()
            flushes += 1
        if started:
            assert self._client is not None
            self._client.finish_weight_update()
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": changed_elems / total_elems,
            "checkpoint_engine/changed_elems": float(changed_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(flushes),
        }
