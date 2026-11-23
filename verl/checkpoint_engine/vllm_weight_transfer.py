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
    *,
    allow_delta_sharded: bool = False,
) -> str | None:
    """Resolve vLLM's weight_transfer_config backend from checkpoint_engine and engine_kwargs."""
    if checkpoint_backend == "delta_sharded" and not allow_delta_sharded:
        raise ValueError(
            "checkpoint_engine.backend='delta_sharded' is no longer used with vLLM; "
            "use checkpoint_engine.backend='vllm_sparse_nccl' instead."
        )

    wtc = (engine_kwargs or {}).get("weight_transfer_config")
    if wtc is not None and not isinstance(wtc, Mapping):
        raise TypeError("rollout.engine_kwargs.vllm.weight_transfer_config must be a mapping")
    configured = wtc.get("backend") if wtc is not None else None

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

    def _fanout(self, method: str, *args: Any) -> None:
        ray.get([getattr(h, method).remote(*args) for h in self.server_handles])

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        if "rank_offset" in init_info:
            offset = int(init_info["rank_offset"])
            payloads = []
            for ws in self.replica_world_sizes:
                payloads.append({**init_info, "rank_offset": offset})
                offset += ws
        elif "num_consumers" in init_info and len(self.server_handles) > 1:
            if len(set(self.replica_world_sizes)) != 1:
                raise ValueError(
                    f"vllm_sharded_rdt requires uniform world sizes across replicas; got {self.replica_world_sizes}"
                )
            n = len(self.server_handles)
            payloads = [{**init_info, "replica_rank": i, "num_replicas": n} for i in range(n)]
        else:
            self._fanout("init_weight_transfer_engine", init_info)
            return
        ray.get([h.init_weight_transfer_engine.remote(p) for h, p in zip(self.server_handles, payloads, strict=True)])

    def shutdown_weight_transfer_engine(self) -> None:
        self._fanout("shutdown_weight_transfer_engine")

    def start_weight_update(self) -> None:
        self._fanout("start_weight_update")

    def update_weights(self, update_info: dict[str, Any] | list[dict[str, Any]]) -> None:
        self._fanout("update_weights", update_info)

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self._fanout("finish_weight_update", weight_version)


def _iter_engine_params(engine: Any, *, shards: bool = False, layered_summon: bool = False):
    params, peft_config = (
        engine.get_per_tensor_param_shard()
        if shards
        else engine.get_per_tensor_param(layered_summon=layered_summon, base_sync_done=True)
    )
    if peft_config is not None:
        raise NotImplementedError(
            "vLLM native weight transfer does not support unmerged LoRA adapters; "
            "set actor_rollout_ref.model.lora.merge=True or checkpoint_engine.backend='naive'."
        )
    return iter(params)


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

        if not self._use_local_shards:
            return [
                ParamMeta(name, t.dtype, tuple(t.shape))
                for name, t in _iter_engine_params(self._engine, layered_summon=self._layered_summon)
            ]

        all_ranks = tuple(range(self._trainer_world_size))
        metas = []
        for name, param in _iter_engine_params(self._engine, shards=True):
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

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        return _iter_engine_params(self._engine, layered_summon=self._layered_summon)

    def iter_local(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, param in _iter_engine_params(self._engine, shards=True):
            layout = _vllm_dtensor_layout(param) if _vllm_dtensor_layout is not None else None
            if layout is None:
                yield name, param
            elif layout[2] is None:
                yield name, param.full_tensor()
            else:
                yield name, param.to_local()


def _dist_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _broadcast_rendezvous_endpoint(rank: int) -> tuple[str, int]:
    endpoint: list[Any] = [None]
    if rank == 0:
        ip = ray.util.get_node_ip_address().strip("[]")
        port, _ = get_free_port(ip)
        endpoint[0] = (ip, int(port))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(endpoint, src=0)
    return endpoint[0]


def _build_trainer_init_info(
    backend: str,
    *,
    rank: int,
    trainer_world_size: int,
    client: VerlRayVLLMWeightSyncClient,
    bucket_size: int,
    trainer_kwargs: dict[str, Any],
) -> Any:
    if backend == "ipc":
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

        return IPCTrainerInitInfo(
            rank=rank,
            packed=bool(trainer_kwargs.get("packed", True)),
            packed_buffer_size_bytes=int(trainer_kwargs.get("packed_buffer_size_bytes", bucket_size)),
        )
    if backend == "nccl":
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        addr, port = _broadcast_rendezvous_endpoint(rank)
        return NCCLTrainerInitInfo(
            master_address=addr,
            master_port=port,
            world_size=1 + client.total_workers,
            rank=rank,
            packed=bool(trainer_kwargs.get("packed", True)),
            packed_buffer_size_bytes=int(trainer_kwargs.get("packed_buffer_size_bytes", bucket_size)),
            packed_num_buffers=int(trainer_kwargs.get("packed_num_buffers", 2)),
            use_load_plan=bool(trainer_kwargs.get("use_load_plan", False)),
        )
    if backend == "sparse_nccl":
        from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseNCCLTrainerInitInfo

        addr, port = _broadcast_rendezvous_endpoint(rank)
        return SparseNCCLTrainerInitInfo(
            master_address=addr, master_port=port, world_size=1 + client.total_workers, rank=rank
        )
    if backend == "sharded_rdt":
        from vllm.distributed.weight_transfer.sharded_rdt_trainer import ShardedRDTTrainerInitInfo

        wpr = client.replica_world_sizes[0] if len(set(client.replica_world_sizes)) == 1 else 0
        ns = trainer_kwargs.get("trainer_actor_namespace")
        if ns is None and ray.is_initialized():
            ns = ray.get_runtime_context().namespace
        extra = {k: v for k, v in trainer_kwargs.items() if k not in {"trainer_actor_namespace", "workers_per_replica"}}
        return ShardedRDTTrainerInitInfo(
            rank=rank,
            num_consumers=client.total_workers,
            workers_per_replica=int(trainer_kwargs.get("workers_per_replica", wpr)),
            trainer_actor_namespace=ns,
            **extra,
        )
    if backend == "nccl_m2n":
        from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo

        addr, port = _broadcast_rendezvous_endpoint(rank)
        extra = dict(trainer_kwargs)
        if extra.get("dst_mesh_dims") is not None:
            extra["dst_mesh_dims"] = tuple(extra["dst_mesh_dims"])
        return M2NTrainerInitInfo(
            master_address=addr,
            master_port=port,
            world_size=trainer_world_size + client.total_workers,
            num_trainer_ranks=trainer_world_size,
            rank=rank,
            **extra,
        )
    raise ValueError(f"Unsupported vLLM native weight transfer backend: {backend!r}")


class _SparsePatchStreamer:
    """Bucket and stream SparseWeightPatch chunks inside one start/finish bracket."""

    def __init__(self, trainer_engine: Any, client: VerlRayVLLMWeightSyncClient, bucket_size: int) -> None:
        self._engine = trainer_engine
        self._client = client
        self._bucket_size = bucket_size
        self._pending: list[Any] = []
        self._pending_bytes = 0
        self._started = False
        self.flushes = 0
        self.changed_elems = 0
        self.wire_bytes = 0

    def push(self, name: str, shape: tuple[int, ...], indices: torch.Tensor, values: torch.Tensor) -> None:
        from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch

        numel = int(indices.numel())
        if numel == 0:
            return
        self.changed_elems += numel
        elem_bytes = torch.int32.itemsize + values.element_size()
        self.wire_bytes += numel * elem_bytes
        for s in range(0, numel, _SPARSE_MAX_ENTRY_ELEMS):
            e = min(s + _SPARSE_MAX_ENTRY_ELEMS, numel)
            patch = SparseWeightPatch(name=name, indices=indices[s:e], values=values[s:e], full_shape=shape)
            pbytes = (e - s) * elem_bytes
            if not self._started:
                self._client.start_weight_update()
                self._started = True
            if self._pending and self._pending_bytes + pbytes > self._bucket_size:
                self._engine.send_weight_chunk(self._pending)
                self._pending.clear()
                self._pending_bytes = 0
                self.flushes += 1
            self._pending.append(patch)
            self._pending_bytes += pbytes

    def finish(self, total_elems: int) -> dict[str, float] | None:
        if self._pending:
            self._engine.send_weight_chunk(self._pending)
            self._pending.clear()
            self.flushes += 1
        if self._started:
            self._client.finish_weight_update()
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": self.changed_elems / total_elems,
            "checkpoint_engine/changed_elems": float(self.changed_elems),
            "checkpoint_engine/payload_mbytes": self.wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(self.flushes),
        }


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
        rollout_dtype: torch.dtype | str = torch.bfloat16,
        batch_gather: int = 32,
        skip_seed: bool = False,
        **trainer_kwargs: Any,
    ) -> None:
        self.bucket_size = int(bucket_size)
        self.is_master = is_master
        self.rebuild_group = rebuild_group
        self.layered_summon = layered_summon
        self.rollout_dtype = getattr(torch, rollout_dtype) if isinstance(rollout_dtype, str) else rollout_dtype
        self.batch_gather = int(batch_gather)
        self.skip_seed = bool(skip_seed)
        self.trainer_kwargs = dict(trainer_kwargs)

        self._client: VerlRayVLLMWeightSyncClient | None = None
        self._target_sig: tuple[tuple[Any, int], ...] | None = None
        self._trainer_engine: Any = None
        self._weight_source: VerlEngineWeightSource | None = None
        self._shard_seeded = False

    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        return {}, {}

    def init_process_group(self, **kwargs: Any) -> None:
        pass

    def configure_targets(self, server_handles: list[Any], replica_world_sizes: list[int]) -> None:
        """Bind target vLLM server actors; re-initializes if the replica set changed."""
        sig = tuple(
            (h._ray_actor_id.hex() if hasattr(getattr(h, "_ray_actor_id", None), "hex") else id(h), int(w))
            for h, w in zip(server_handles, replica_world_sizes, strict=True)
        )
        if sig == self._target_sig and self._client is not None:
            return
        self._shutdown_trainer_engine()
        self._client = VerlRayVLLMWeightSyncClient(server_handles, replica_world_sizes)
        self._target_sig = sig

    def _shutdown_trainer_engine(self) -> None:
        if self._trainer_engine is None:
            return
        rank, _ = _dist_rank_world()
        try:
            self._trainer_engine.shutdown()
        finally:
            self._trainer_engine = None
        if rank == 0 and self._client is not None:
            self._client.shutdown_weight_transfer_engine()

    def finalize(self) -> None:
        if self.rebuild_group:
            self._shutdown_trainer_engine()

    def _ensure_trainer_engine(self, engine: Any) -> Any:
        if self._trainer_engine is not None:
            return self._trainer_engine
        if self._client is None:
            raise RuntimeError("configure_targets() must be called before send_weights()")

        from vllm.distributed.weight_transfer import WeightTransferTrainerFactory

        rank, world_size = _dist_rank_world()
        source = None
        if self.vllm_backend != "sparse_nccl":
            if self._weight_source is None or self._weight_source._engine is not engine:
                self._weight_source = VerlEngineWeightSource(
                    engine,
                    layered_summon=self.layered_summon,
                    use_local_shards=(self.vllm_backend == "nccl_m2n"),
                    trainer_world_size=world_size,
                )
            source = self._weight_source

        init_info = _build_trainer_init_info(
            self.vllm_backend,
            rank=rank,
            trainer_world_size=world_size,
            client=self._client,
            bucket_size=self.bucket_size,
            trainer_kwargs=self.trainer_kwargs,
        )
        self._trainer_engine = WeightTransferTrainerFactory.trainer_init(
            init_info=init_info, client=self._client, source=source
        )
        return self._trainer_engine

    def _send_sparse_weights(self, engine: Any, trainer_engine: Any) -> dict[str, float] | None:
        from verl.checkpoint_engine.delta_checkpoint_engine import _GatherQueue

        assert self._client is not None
        rank, _ = _dist_rank_world()
        is_sender = rank == 0
        streamer = _SparsePatchStreamer(trainer_engine, self._client, self.bucket_size)
        total_elems = 0

        if not self._shard_seeded:
            if not self.skip_seed:
                for name, tensor in _iter_engine_params(engine, layered_summon=self.layered_summon):
                    tensor = tensor.detach()
                    if not tensor.is_floating_point():
                        continue
                    if tensor.dtype != self.rollout_dtype:
                        tensor = tensor.to(self.rollout_dtype)
                    if not is_sender:
                        del tensor
                        continue
                    flat = tensor.contiguous().reshape(-1)
                    numel, shape = int(flat.numel()), tuple(tensor.shape)
                    total_elems += numel
                    for s in range(0, numel, _SPARSE_MAX_ENTRY_ELEMS):
                        e = min(s + _SPARSE_MAX_ENTRY_ELEMS, numel)
                        streamer.push(name, shape, torch.arange(s, e, dtype=torch.int32, device=flat.device), flat[s:e])
            engine.prime_delta_snapshots()
            self._shard_seeded = True
            return streamer.finish(total_elems) if is_sender else None

        weights, _ = engine.get_per_tensor_param_delta_shard()

        def _consume(name, _dtype_str, full_shape, full_numel, aidx, aval):
            nonlocal total_elems
            total_elems += int(full_numel)
            if aidx is not None and aval is not None and aidx.numel() > 0:
                streamer.push(name, tuple(full_shape), aidx.contiguous().to(dtype=torch.int32), aval.contiguous())

        gq = _GatherQueue(self.batch_gather, self.bucket_size, is_sender, _consume)
        for slots, dtype_str, counts, hf_idx, hf_val, pg in weights:
            gq.put(pg, slots, dtype_str, counts, hf_idx, hf_val)
        gq.flush_all()
        return streamer.finish(total_elems) if is_sender else None

    async def send_weights(self, engine: Any, global_steps: int | None = None) -> dict[str, float] | None:
        t0 = time.perf_counter()
        trainer_engine = self._ensure_trainer_engine(engine)
        if self.vllm_backend == "sparse_nccl":
            return self._send_sparse_weights(engine, trainer_engine)
        trainer_engine.send_weights()
        return {"checkpoint_engine/sync_time_s": time.perf_counter() - t0} if self.is_master else None

    async def receive_weights(self, global_steps: int | None = None):
        raise NotImplementedError("vLLM native weight transfer delivers weights directly to vLLM workers")


for _ce_name, _wt_name in VLLM_NATIVE_BACKENDS.items():
    CheckpointEngineRegistry.register(_ce_name)(
        type(
            f"VLLM{_wt_name.upper()}CheckpointEngine",
            (VLLMWeightTransferCheckpointEngine,),
            {"vllm_backend": _wt_name, "is_colocated": _wt_name == "ipc"},
        )
    )
