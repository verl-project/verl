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

"""P2P transfer planning utilities, interoperable with the Miles P2P protocol.

Transfer planning follows Miles heuristics; implementation is built around verl
``RolloutReplica`` handles instead of Miles rollout Ray actors.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol

import ray
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class P2PRolloutTopology:
    """Rollout-side topology for P2P transfer planning."""

    engine_count: int
    gpus_per_engine: int
    pipeline_parallel_size: int = 1

    @property
    def total_gpus(self) -> int:
        return self.engine_count * self.gpus_per_engine


@dataclasses.dataclass
class TransferTaskP2PMeta:
    """Specifies a rollout engine rank to connect to."""

    engine_ind: int
    engine_rank: int


@dataclasses.dataclass
class RemoteWeightInfo:
    """Remote weight registration info for one rollout session."""

    session_id: str
    weights_info: dict[str, tuple[int, int, int]]


class P2PRolloutReplica(Protocol):
    async def get_remote_instance_transfer_engine_info(self, rank: int) -> Any: ...

    async def get_parallelism_info(self, rank: int) -> Any: ...

    async def get_server_info(self) -> dict[str, Any]: ...


def resolve_trainer_parallelism() -> tuple[int, int, int, int]:
    """Return ``(pp_rank, pp_size, gathered_dp_rank, gathered_dp_size)``."""
    try:
        from megatron.core import parallel_state as mpu
    except ImportError as exc:
        raise RuntimeError("Megatron parallel_state is required for P2P transfer planning") from exc

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for P2P transfer planning")

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    world_size = dist.get_world_size()
    gathered_dp_size = world_size // pp_size

    pp_group = mpu.get_pipeline_model_parallel_group()
    my_column_id = min(dist.get_process_group_ranks(pp_group))
    all_column_ids = [None] * world_size
    dist.all_gather_object(all_column_ids, my_column_id)
    sorted_columns = sorted(set(all_column_ids))
    gathered_dp_rank = sorted_columns.index(my_column_id)
    return pp_rank, pp_size, gathered_dp_rank, gathered_dp_size


class RemoteTransferPlan:
    """Plans trainer -> rollout P2P connections (Miles-compatible heuristics)."""

    def __init__(self, rollout_topology: P2PRolloutTopology) -> None:
        if rollout_topology.pipeline_parallel_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not supported for P2P yet.")

        _, _, self._gathered_dp_rank, self._gathered_dp_size = resolve_trainer_parallelism()
        self._rollout_num_gpu_per_engine = rollout_topology.gpus_per_engine
        self._rollout_engine_count = rollout_topology.engine_count
        self._rollout_num_gpus = rollout_topology.total_gpus

    @property
    def gathered_dp_rank(self) -> int:
        return self._gathered_dp_rank

    @property
    def is_source(self) -> bool:
        return self._gathered_dp_rank < self._rollout_num_gpus

    def plan_p2p(self) -> list[TransferTaskP2PMeta]:
        all_targets = [
            (engine_idx, engine_rank)
            for engine_idx in range(self._rollout_engine_count)
            for engine_rank in range(self._rollout_num_gpu_per_engine)
        ]
        assignments: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

        p2p_count = 0
        for source_rank, (_, target) in zip(range(self._gathered_dp_size), enumerate(all_targets), strict=False):
            p2p_count += 1
            engine_idx, engine_rank = target
            assignments[source_rank][engine_rank].append(engine_idx)

        def count_engine_index_assignments(engine_rank: int) -> list[int]:
            return [len(assignments[source][engine_rank]) for source in range(self._gathered_dp_size)]

        cur_source_index = 0
        if p2p_count < len(all_targets):
            for target in all_targets[p2p_count:]:
                engine_idx, engine_rank = target
                counted = count_engine_index_assignments(engine_rank)
                if max(counted) > 0:
                    _, select_source = min((val, idx) for (idx, val) in enumerate(counted) if val > 0)
                else:
                    select_source = cur_source_index % self._gathered_dp_size
                    cur_source_index += 1
                assignments[select_source][engine_rank].append(engine_idx)

        transfer_tasks: list[TransferTaskP2PMeta] = []
        for engine_rank, engine_indices in assignments[self._gathered_dp_rank].items():
            for engine_ind in engine_indices:
                transfer_tasks.append(
                    TransferTaskP2PMeta(
                        engine_ind=engine_ind,
                        engine_rank=engine_rank,
                    )
                )
        return transfer_tasks


class P2PTransferManager:
    """Async thread-pool executor for RDMA writes."""

    def __init__(self, num_workers: int = 8, transfer_timeout: float = 30.0) -> None:
        self.num_workers = num_workers
        self.transfer_timeout = transfer_timeout
        self.executor: ThreadPoolExecutor | None = None
        self.transfer_futures: list[Future] = []

    def ensure_started(self) -> None:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def submit(self, fn: Callable, *args) -> None:
        self.ensure_started()
        self.transfer_futures.append(self.executor.submit(fn, *args))

    def submit_returning_future(self, fn: Callable, *args) -> Future:
        self.ensure_started()
        future = self.executor.submit(fn, *args)
        self.transfer_futures.append(future)
        return future

    def wait_futures(self, futures: Sequence[Future]) -> None:
        for future in futures:
            future.result(timeout=self.transfer_timeout)

    def wait_transfers(self) -> None:
        for future in self.transfer_futures:
            try:
                future.result(timeout=self.transfer_timeout)
            except Exception as exc:
                logger.error("P2P transfer future failed: %s", exc)
                raise
        self.transfer_futures.clear()


def _json_safe_value(value: Any) -> Any:
    """Coerce rollout metadata values to Ray-pickle-safe primitives."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return None


def filter_server_args_dict(data_dict: dict[str, Any]) -> dict[str, Any]:
    """Keep only ServerArgs fields with JSON/pickle-safe values for cross-node transfer."""
    import dataclasses as dc

    from sglang.srt.server_args import ServerArgs

    valid_fields = {f.name for f in dc.fields(ServerArgs)}
    return {key: _json_safe_value(value) for key, value in data_dict.items() if key in valid_fields}


def create_server_args_from_dict(data_dict: dict[str, Any]):
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(**data_dict)


def register_cpu_memory(params_dict: dict[str, torch.Tensor], transfer_engine) -> dict[str, tuple[int, int, int]]:
    weight_dict: dict[str, tuple[int, int, int]] = {}
    for name, cpu_tensor in params_dict.items():
        addr = cpu_tensor.data_ptr()
        size = cpu_tensor.numel() * cpu_tensor.element_size()
        ret = transfer_engine.register_memory(addr, size)
        if ret != 0:
            raise RuntimeError(f"register CPU memory failed for weight {name}, error: {ret}")
        weight_dict[name] = (addr, cpu_tensor.numel(), cpu_tensor.element_size())
    return weight_dict


def unregister_cpu_memory(weight_memory_registry: dict[str, tuple[int, int, int]], transfer_engine) -> None:
    """Unregister Mooncake-registered CPU buffers before freeing the P2P replica."""
    if not weight_memory_registry:
        return

    for name, (addr, _numel, _ele_size) in list(weight_memory_registry.items()):
        ret = transfer_engine.unregister_memory(addr)
        if ret != 0:
            logger.warning("unregister CPU memory failed for weight %s, error: %s", name, ret)
    weight_memory_registry.clear()


def resolve_mooncake_transfer_engine_settings() -> tuple[str, str]:
    """Return Mooncake protocol and IB device filter (aligned with SGLang envs)."""
    protocol = os.environ.get("MOONCAKE_PROTOCOL", "rdma")
    if os.environ.get("MC_FORCE_TCP") == "1":
        return protocol, ""
    return protocol, os.environ.get("MOONCAKE_DEVICE", "")


def create_transfer_engine():
    from mooncake.engine import TransferEngine

    transfer_engine = TransferEngine()
    local_ip = ray._private.services.get_node_ip_address()
    protocol, device_name = resolve_mooncake_transfer_engine_settings()
    ret = transfer_engine.initialize(local_ip, "P2PHANDSHAKE", protocol, device_name)
    if ret != 0:
        raise RuntimeError(
            f"TransferEngine initialize failed with ret={ret} (protocol={protocol!r}, device_name={device_name!r})"
        )
    return transfer_engine


async def query_remote_weight_infos(
    replicas: Sequence[P2PRolloutReplica],
    targets: Sequence[TransferTaskP2PMeta],
) -> tuple[dict[str, tuple[Any, Any]], dict[tuple[int, int], str], dict[str, Any]]:
    """Query rollout replicas for transfer metadata (async, Verl replica API)."""
    remote_weight_infos_by_session_id: dict[str, tuple[Any, Any]] = {}
    targets_to_session_id: dict[tuple[int, int], str] = {}
    session_id_to_server_args: dict[str, Any] = {}
    targets_to_query = {(target.engine_ind, target.engine_rank) for target in targets}

    for engine_ind, engine_rank in targets_to_query:
        replica = replicas[engine_ind]
        transfer_info = await replica.get_remote_instance_transfer_engine_info(engine_rank)
        parallelism_info = await replica.get_parallelism_info(engine_rank)
        if transfer_info is None:
            raise RuntimeError(f"missing transfer engine info for replica={engine_ind} rank={engine_rank}")
        if parallelism_info is None:
            raise RuntimeError(f"missing parallelism config for replica={engine_ind} rank={engine_rank}")

        session_id, weights_info = transfer_info
        assert session_id is not None, f"Failed to get session id from rollout replica {engine_ind} rank {engine_rank}"
        server_info = await replica.get_server_info()
        # Store plain dicts only: ServerArgs objects may pickle references to
        # transformers_modules (trust_remote_code) and fail on trainer workers.
        session_id_to_server_args[session_id] = filter_server_args_dict(server_info)
        remote_weight_infos_by_session_id[session_id] = (weights_info, parallelism_info)
        targets_to_session_id[(engine_ind, engine_rank)] = session_id

    return remote_weight_infos_by_session_id, targets_to_session_id, session_id_to_server_args


def build_rollout_topology_from_replicas(replicas: Sequence[Any]) -> P2PRolloutTopology:
    if not replicas:
        raise ValueError("At least one rollout replica is required for P2P")
    gpus_per_engine = replicas[0].world_size
    pp_size = getattr(replicas[0].config, "pipeline_model_parallel_size", 1)
    for replica in replicas[1:]:
        if replica.world_size != gpus_per_engine:
            raise NotImplementedError("Heterogeneous replica world sizes are not supported for P2P yet")
    return P2PRolloutTopology(
        engine_count=len(replicas),
        gpus_per_engine=gpus_per_engine,
        pipeline_parallel_size=pp_size,
    )


def serialize_p2p_rollout_metadata(
    *,
    model_path: str,
    rollout_topology: P2PRolloutTopology,
    engine_kwargs: dict[str, Any],
    remote_weight_infos_by_session_id: dict[str, tuple[Any, Any]],
    targets_to_session_id: dict[tuple[int, int], str],
    session_id_to_server_args: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_path": model_path,
        "rollout_topology": dataclasses.asdict(rollout_topology),
        "engine_kwargs": engine_kwargs,
        "remote_weight_infos_by_session_id": remote_weight_infos_by_session_id,
        "targets_to_session_id": targets_to_session_id,
        "session_id_to_server_args": session_id_to_server_args,
    }


def iter_named_tensor_buckets(
    weights: Iterator[tuple[str, torch.Tensor]],
    bucket_bytes: int,
    *,
    clone_tensors: bool = False,
) -> Generator[list[tuple[str, torch.Tensor]], None, None]:
    """Group HF weight tensors into byte-bounded buckets (Miles-style export batching)."""
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket: list[tuple[str, torch.Tensor]] = []
    current_size = 0
    for name, tensor in weights:
        tensor_size = tensor.element_size() * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield current_bucket
            stored = tensor.clone() if clone_tensors else tensor
            current_bucket = [(name, stored)]
            current_size = tensor_size
        else:
            stored = tensor.clone() if clone_tensors else tensor
            current_bucket.append((name, stored))
            current_size += tensor_size

    if current_bucket:
        yield current_bucket


def resolve_rollout_model_path(model_config: DictConfig) -> str:
    """Resolve the rollout HF model path from the driver-side replica config."""
    from omegaconf import DictConfig, OmegaConf

    from verl.utils.fs import copy_to_local

    if not isinstance(model_config, DictConfig):
        raise TypeError(f"resolve_rollout_model_path expects DictConfig, got {type(model_config).__name__}")

    path = OmegaConf.select(model_config, "path", default=None)
    if not path:
        raise RuntimeError("P2P weight sync requires actor_rollout_ref.model.path")

    use_shm = OmegaConf.select(model_config, "use_shm", default=False)
    return copy_to_local(path, use_shm=use_shm)


async def collect_p2p_rollout_metadata(
    replicas: Sequence[P2PRolloutReplica],
    *,
    model_path: str,
    engine_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    topology = build_rollout_topology_from_replicas(replicas)
    all_targets = [
        TransferTaskP2PMeta(engine_ind=engine_ind, engine_rank=engine_rank)
        for engine_ind, replica in enumerate(replicas)
        for engine_rank in range(replica.world_size)
    ]
    remote_infos, targets_to_session_id, session_id_to_server_args = await query_remote_weight_infos(
        replicas, all_targets
    )
    return serialize_p2p_rollout_metadata(
        model_path=model_path,
        rollout_topology=topology,
        engine_kwargs=engine_kwargs or {},
        remote_weight_infos_by_session_id=remote_infos,
        targets_to_session_id=targets_to_session_id,
        session_id_to_server_args=session_id_to_server_args,
    )
