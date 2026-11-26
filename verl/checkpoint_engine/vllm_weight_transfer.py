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
import math
import time
from collections.abc import Iterator, Mapping
from functools import cached_property
from typing import Any

import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)

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
    if expected is not None and configured not in (None, expected):
        raise ValueError(
            f"checkpoint_engine.backend={checkpoint_backend!r} requires vLLM weight_transfer_config "
            f"backend={expected!r}, got {configured!r}"
        )
    return expected or configured


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
            return self._fanout("init_weight_transfer_engine", init_info)
        ray.get([h.init_weight_transfer_engine.remote(p) for h, p in zip(self.server_handles, payloads, strict=True)])

    def shutdown_weight_transfer_engine(self) -> None:
        refs = [h.shutdown_weight_transfer_engine.remote() for h in self.server_handles]
        first_err: Exception | None = None
        for ref in refs:
            try:
                ray.get([ref])
            except ray.exceptions.RayActorError as exc:
                if first_err is None:
                    first_err = exc
        if first_err is not None:
            raise first_err

    def start_weight_update(self) -> None:
        self._fanout("start_weight_update")

    def start_draft_weight_update(self) -> None:
        self._fanout("start_draft_weight_update")

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


def _shard_spec_layout(
    spec: Any, trainer_world_size: int, local_numel: int | None = None
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Map a verl ShardSpec to vLLM ParamMeta layout kwargs and this rank's local tensor shape."""
    from vllm.distributed.weight_transfer.base import Replicate, Shard

    if getattr(spec, "place", None) is not None:
        raise NotImplementedError(f"vllm_nccl_m2n does not support ShardSpec.place={spec.place!r}")
    if getattr(spec, "to_hf_chunk", None) is not None or getattr(spec, "hf_slots", None) is not None:
        raise NotImplementedError(
            "vllm_nccl_m2n does not support ShardSpec converter/slot exports (to_hf_chunk/hf_slots)"
        )
    if not getattr(spec, "contributes", True):
        raise NotImplementedError("vllm_nccl_m2n does not support ShardSpec with contributes=False")

    full_shape = tuple(int(d) for d in spec.full_shape)
    if spec.mesh is None or not spec.placements:
        full_numel = math.prod(full_shape)
        if local_numel is not None and local_numel != full_numel:
            raise NotImplementedError(
                f"vllm_nccl_m2n does not support unmeshed local shards with size {local_numel} != "
                f"full_shape {full_shape} ({full_numel} elements)"
            )
        return {"owners": tuple(range(trainer_world_size))}, full_shape

    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    from torch.distributed.tensor.placement_types import Replicate as TorchReplicate
    from torch.distributed.tensor.placement_types import Shard as TorchShard

    owners = tuple(int(r) for r in spec.mesh.mesh.reshape(-1).tolist())
    mesh_shape = tuple(int(d) for d in spec.mesh.shape)
    vllm_placements = []
    for p in spec.placements:
        # Exact type checks: _StridedShard subclasses TorchShard and carries split_factor semantics
        # that vLLM's plain Shard(dim) cannot represent without corrupting weights.
        if type(p) is TorchShard:
            vllm_placements.append(Shard(p.dim))
        elif type(p) is TorchReplicate:
            vllm_placements.append(Replicate())
        else:
            raise NotImplementedError(f"Unsupported DTensor placement for vllm_nccl_m2n: {p!r}")
    local_shape, _ = compute_local_shape_and_global_offset(full_shape, spec.mesh, spec.placements)
    extra: dict[str, Any] = {
        "owners": owners,
        "mesh_shape": mesh_shape,
        "placements": None if all(isinstance(p, Replicate) for p in vllm_placements) else tuple(vllm_placements),
    }
    return extra, tuple(int(d) for d in local_shape)


class VerlEngineWeightSource(_WeightSourceBase):
    """WeightSource adapter over a verl training engine's HF parameter export."""

    def __init__(
        self, engine: Any, *, layered_summon: bool = False, use_local_shards: bool = False, trainer_world_size: int = 1
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
        for entry in _iter_engine_params(self._engine, shards=True):
            if len(entry) == 3:
                name, local_tensor, spec = entry
                extra, _ = _shard_spec_layout(spec, self._trainer_world_size, local_numel=int(local_tensor.numel()))
                metas.append(ParamMeta(name, local_tensor.dtype, tuple(int(d) for d in spec.full_shape), **extra))
            else:
                name, p = entry
                layout = _vllm_dtensor_layout(p) if _vllm_dtensor_layout is not None else None
                extra = (
                    {"owners": all_ranks}
                    if layout is None
                    else {"owners": layout[0], "mesh_shape": layout[1], "placements": layout[2]}
                )
                metas.append(ParamMeta(name, p.dtype, tuple(p.shape), **extra))
        return metas

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        return _iter_engine_params(self._engine, layered_summon=self._layered_summon)

    def iter_local(self) -> Iterator[tuple[str, torch.Tensor]]:
        for entry in _iter_engine_params(self._engine, shards=True):
            if len(entry) == 3:
                name, local_tensor, spec = entry
                _, local_shape = _shard_spec_layout(
                    spec, self._trainer_world_size, local_numel=int(local_tensor.numel())
                )
                yield name, local_tensor.reshape(local_shape).contiguous()
            else:
                name, p = entry
                layout = _vllm_dtensor_layout(p) if _vllm_dtensor_layout is not None else None
                yield name, p if layout is None else (p.full_tensor() if layout[2] is None else p.to_local())


def _dist_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _broadcast_rendezvous_endpoint(rank: int) -> tuple[str, int]:
    endpoint: list[Any] = [None]
    if rank == 0:
        ip = ray.util.get_node_ip_address().strip("[]")
        endpoint[0] = (ip, int(get_free_port(ip)[0]))
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
    kw = dict(trainer_kwargs)
    if backend == "ipc":
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

        return IPCTrainerInitInfo(rank=rank, **{"packed": True, "packed_buffer_size_bytes": bucket_size, **kw})
    if backend == "nccl":
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        addr, port = _broadcast_rendezvous_endpoint(rank)
        defaults = {"packed": True, "packed_buffer_size_bytes": bucket_size, "packed_num_buffers": 2}
        return NCCLTrainerInitInfo(
            master_address=addr, master_port=port, world_size=1 + client.total_workers, rank=rank, **{**defaults, **kw}
        )
    if backend == "sparse_nccl":
        from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseNCCLTrainerInitInfo

        addr, port = _broadcast_rendezvous_endpoint(rank)
        return SparseNCCLTrainerInitInfo(
            master_address=addr, master_port=port, world_size=1 + client.total_workers, rank=rank, **kw
        )
    if backend == "sharded_rdt":
        from vllm.distributed.weight_transfer.sharded_rdt_trainer import ShardedRDTTrainerInitInfo

        defaults = {
            "workers_per_replica": client.replica_world_sizes[0] if len(set(client.replica_world_sizes)) == 1 else 0,
            "trainer_actor_namespace": ray.get_runtime_context().namespace if ray.is_initialized() else None,
        }
        return ShardedRDTTrainerInitInfo(rank=rank, num_consumers=client.total_workers, **{**defaults, **kw})
    if backend == "nccl_m2n":
        from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo

        addr, port = _broadcast_rendezvous_endpoint(rank)
        if kw.get("dst_mesh_dims") is not None:
            kw["dst_mesh_dims"] = tuple(kw["dst_mesh_dims"])
        return M2NTrainerInitInfo(
            master_address=addr,
            master_port=port,
            world_size=trainer_world_size + client.total_workers,
            num_trainer_ranks=trainer_world_size,
            rank=rank,
            **kw,
        )
    raise ValueError(f"Unsupported vLLM native weight transfer backend: {backend!r}")


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
        self._shard_seeded = False

    def _shutdown_trainer_engine(self) -> None:
        if self._trainer_engine is None:
            return
        rank, _ = _dist_rank_world()
        trainer_engine = self._trainer_engine
        self._trainer_engine = None
        try:
            trainer_engine.shutdown()
        except Exception as exc:
            logger.warning("Failed to shut down vLLM trainer weight transfer engine cleanly: %s", exc)
        if rank == 0 and self._client is not None:
            try:
                self._client.shutdown_weight_transfer_engine()
            except ray.exceptions.RayActorError as exc:
                logger.warning("Ignoring dead vLLM server actor during weight transfer shutdown: %s", exc)

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
        from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch

        from verl.checkpoint_engine.delta_checkpoint_engine import _GatherQueue

        assert self._client is not None
        is_sender = _dist_rank_world()[0] == 0
        pending: list[Any] = []
        pending_bytes = flushes = changed_elems = total_elems = wire_bytes = 0
        started = False

        def _push(name: str, shape: tuple[int, ...], indices: torch.Tensor, values: torch.Tensor) -> None:
            nonlocal pending_bytes, flushes, changed_elems, wire_bytes, started
            n = int(indices.numel())
            if n == 0:
                return
            changed_elems += n
            eb = torch.int32.itemsize + values.element_size()
            wire_bytes += n * eb
            for s in range(0, n, _SPARSE_MAX_ENTRY_ELEMS):
                e = min(s + _SPARSE_MAX_ENTRY_ELEMS, n)
                pb = (e - s) * eb
                if not started:
                    self._client.start_weight_update()
                    started = True
                if pending and pending_bytes + pb > self.bucket_size:
                    trainer_engine.send_weight_chunk(pending)
                    pending.clear()
                    pending_bytes = 0
                    flushes += 1
                pending.append(SparseWeightPatch(name=name, indices=indices[s:e], values=values[s:e], full_shape=shape))
                pending_bytes += pb

        def _flush_pending() -> None:
            nonlocal pending_bytes, flushes
            if is_sender and pending:
                trainer_engine.send_weight_chunk(pending)
                pending.clear()
                pending_bytes = 0
                flushes += 1

        try:
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
                            _push(name, shape, torch.arange(s, e, dtype=torch.int32, device=flat.device), flat[s:e])
                _flush_pending()
                engine.prime_delta_snapshots()
                self._shard_seeded = True
            else:
                weights, _ = engine.get_per_tensor_param_delta_shard()

                def _consume(name, _dtype_str, full_shape, full_numel, aidx, aval):
                    nonlocal total_elems
                    total_elems += int(full_numel)
                    if aidx is not None and aval is not None and aidx.numel() > 0:
                        _push(name, tuple(full_shape), aidx.contiguous().to(dtype=torch.int32), aval.contiguous())

                gq = _GatherQueue(self.batch_gather, self.bucket_size, is_sender, _consume)
                for slots, dtype_str, counts, hf_idx, hf_val, pg in weights:
                    gq.put(pg, slots, dtype_str, counts, hf_idx, hf_val)
                gq.flush_all()
                _flush_pending()
        finally:
            if is_sender and started:
                self._client.finish_weight_update()

        if not is_sender:
            return None
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": changed_elems / total_elems,
            "checkpoint_engine/changed_elems": float(changed_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(flushes),
        }

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
