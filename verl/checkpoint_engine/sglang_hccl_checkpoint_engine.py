# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Direct HCCL weight synchronization from Megatron to SGLang."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import ray
import requests
import torch

from verl.utils.net_utils import get_free_port
from verl.utils.ray_utils import auto_await

from .base import CheckpointEngine, CheckpointEngineManager, CheckpointEngineRegistry

LOGGER = logging.getLogger(__name__)
BACKEND_NAME = "sglang_hccl"
DEFAULT_GROUP_NAME = "verl_sglang_hccl"


def iter_weight_buckets(
    weights: Iterable[tuple[str, torch.Tensor]], bucket_size: int
) -> Generator[list[tuple[str, torch.Tensor]], None, None]:
    """Group complete named tensors without splitting a model parameter."""
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_bytes = 0
    for name, tensor in weights:
        tensor_bytes = tensor.nbytes
        if tensor_bytes > bucket_size:
            raise ValueError(
                f"Weight {name} requires {tensor_bytes} bytes, exceeding the {bucket_size}-byte synchronization bucket"
            )
        if bucket and bucket_bytes + tensor_bytes > bucket_size:
            yield bucket
            bucket = []
            bucket_bytes = 0
        bucket.append((name, tensor))
        bucket_bytes += tensor_bytes

    if bucket:
        yield bucket


def deranged_rollout_indices(world_size: int) -> list[int]:
    """Map every trainer rank to a rollout rank on a different device."""
    if world_size < 2:
        raise ValueError("Direct colocated HCCL synchronization requires at least two ranks")
    return list(range(1, world_size)) + [0]


def _post_json(base_url: str, endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(f"{base_url}/{endpoint}", json=payload, timeout=timeout)
    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        result = {"message": response.text}
    if result is None:
        result = {}
    elif not isinstance(result, dict):
        result = {"message": result}
    if not response.ok:
        raise RuntimeError(
            f"SGLang {endpoint} failed at {base_url} with HTTP {response.status_code}: {result.get('message', result)}"
        )
    if result.get("success") is False:
        raise RuntimeError(f"SGLang {endpoint} failed at {base_url}: {result.get('message', result)}")
    return result


@CheckpointEngineRegistry.register(BACKEND_NAME)
class SGLangHCCLCheckpointEngine(CheckpointEngine):
    """Broadcast flattened HF weight buckets directly into SGLang schedulers."""

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        group_name: str = DEFAULT_GROUP_NAME,
        process_group_backend: str = "hccl",
        request_timeout: float = 3600,
    ) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.group_name = group_name
        self.process_group_backend = process_group_backend
        self.request_timeout = request_timeout
        self.rank: int | None = None
        self.world_size: int | None = None
        self.process_group = None
        self.rollout_endpoints: list[str] = []
        self.master_metadata: dict[str, Any] | None = None

    def prepare(self) -> dict[str, Any]:
        if self.master_metadata is None:
            master_address = ray.util.get_node_ip_address().strip("[]")
            master_port, _ = get_free_port(master_address)
            self.master_metadata = {
                "master_address": master_address,
                "master_port": master_port,
            }
        return self.master_metadata

    @classmethod
    def build_topology(cls, *args, **kwargs):
        raise NotImplementedError("SGLangHCCLCheckpointEngine uses its direct SGLang manager")

    def set_rollout_endpoints(self, endpoints: list[str]) -> None:
        self.rollout_endpoints = list(endpoints)

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        master_address: str,
        master_port: int,
        group_name: str,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        if rank != 0:
            raise ValueError(f"Trainer sender rank must be 0, got {rank}")
        if self.process_group is not None:
            return

        from sglang.srt.utils import init_custom_process_group

        self.process_group = init_custom_process_group(
            backend=self.process_group_backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=self.group_name,
        )

    def finalize(self) -> None:
        if self.process_group is not None:
            torch.distributed.destroy_process_group(self.process_group)
            self.process_group = None

    def _broadcast_bucket(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        global_steps: int | None,
        is_last: bool,
    ) -> int:
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor = bucket.get_flattened_tensor()
        payload = {
            "names": [name for name, _ in named_tensors],
            "dtypes": [str(tensor.dtype).removeprefix("torch.") for _, tensor in named_tensors],
            "shapes": [list(tensor.shape) for _, tensor in named_tensors],
            "group_name": self.group_name,
            "load_format": "flattened_bucket",
            "flush_cache": is_last,
            "weight_version": None if global_steps is None else str(global_steps),
        }

        with ThreadPoolExecutor(max_workers=len(self.rollout_endpoints)) as executor:
            futures = [
                executor.submit(
                    _post_json,
                    endpoint,
                    "update_weights_from_distributed",
                    payload,
                    self.request_timeout,
                )
                for endpoint in self.rollout_endpoints
            ]
            torch.distributed.broadcast(flattened_tensor, src=0, group=self.process_group)
            for future in futures:
                future.result()

        return flattened_tensor.nbytes

    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ) -> None:
        if self.rank is None:
            raise RuntimeError("Direct SGLang HCCL process group is not initialized")

        if not self.rollout_endpoints:
            raise RuntimeError("No SGLang rollout endpoints were configured")

        started = time.monotonic()
        current_bucket: list[tuple[str, torch.Tensor]] | None = None
        bucket_count = 0
        tensor_count = 0
        total_bytes = 0
        buckets = iter_weight_buckets(weights, self.bucket_size)
        for next_bucket in buckets:
            if current_bucket is not None:
                total_bytes += self._broadcast_bucket(current_bucket, global_steps, is_last=False)
                bucket_count += 1
                tensor_count += len(current_bucket)
            current_bucket = next_bucket

        if current_bucket is not None:
            total_bytes += self._broadcast_bucket(current_bucket, global_steps, is_last=True)
            bucket_count += 1
            tensor_count += len(current_bucket)

        elapsed = time.monotonic() - started
        print(
            "SGLang direct HCCL weight sync complete: "
            f"step={global_steps}, tensors={tensor_count}, buckets={bucket_count}, "
            f"bytes={total_bytes}, seconds={elapsed:.3f}",
            flush=True,
        )

    async def receive_weights(self, global_steps: int | None = None):
        raise RuntimeError("SGLang receives this backend directly through HCCL")
        yield  # pragma: no cover


class SGLangHCCLCheckpointEngineManager(CheckpointEngineManager):
    """Coordinate a persistent trainer-to-SGLang HCCL process group."""

    def __init__(self, config, actor_wg, replicas) -> None:
        super().__init__(config=config, actor_wg=actor_wg, replicas=replicas)
        if self.backend != BACKEND_NAME:
            raise ValueError(f"This manager requires checkpoint backend {BACKEND_NAME!r}")

        engine_kwargs = config.engine_kwargs.get(BACKEND_NAME, {})
        self.group_name = engine_kwargs.get("group_name", DEFAULT_GROUP_NAME)
        self.process_group_backend = engine_kwargs.get("process_group_backend", "hccl")
        self.request_timeout = float(engine_kwargs.get("request_timeout", 3600))
        self._group_initialized = False
        self._server_urls = [f"http://{replica.server_address}" for replica in replicas]

    async def _post_all(self, endpoint: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return await asyncio.gather(
            *[asyncio.to_thread(_post_json, url, endpoint, payload, self.request_timeout) for url in self._server_urls]
        )

    async def _initialize_process_group(self) -> None:
        if self._group_initialized:
            return

        trainer = self.actor_wg
        trainer_world_size = trainer.world_size
        if len(self.replicas) != trainer_world_size:
            raise ValueError("Direct colocated HCCL synchronization requires one rollout replica per trainer rank")
        if any(replica.world_size != 1 for replica in self.replicas):
            raise ValueError("Direct colocated HCCL synchronization requires rollout tensor parallel size 1")

        rollout_by_trainer_rank = deranged_rollout_indices(trainer_world_size)
        metadata = ray.get(trainer.execute_checkpoint_engine(["prepare"] * trainer_world_size))
        if any(item is None for item in metadata):
            raise RuntimeError("Every trainer rank must provide HCCL rendezvous metadata")

        group_names = [f"{self.group_name}_{trainer_rank}" for trainer_rank in range(trainer_world_size)]
        endpoints_by_trainer_rank = [[self._server_urls[rollout_rank]] for rollout_rank in rollout_by_trainer_rank]

        ray.get(
            trainer.execute_checkpoint_engine(
                ["set_rollout_endpoints"] * trainer_world_size,
                endpoints_by_trainer_rank,
            )
        )

        init_refs = trainer.execute_checkpoint_engine(
            method=["init_process_group"] * trainer_world_size,
            rank=[0] * trainer_world_size,
            world_size=[2] * trainer_world_size,
            master_address=[item["master_address"] for item in metadata],
            master_port=[item["master_port"] for item in metadata],
            group_name=group_names,
        )

        trainer_by_rollout_rank = [0] * trainer_world_size
        for trainer_rank, rollout_rank in enumerate(rollout_by_trainer_rank):
            trainer_by_rollout_rank[rollout_rank] = trainer_rank

        init_payloads = []
        for rollout_rank, trainer_rank in enumerate(trainer_by_rollout_rank):
            item = metadata[trainer_rank]
            init_payloads.append(
                {
                    "master_address": item["master_address"],
                    "master_port": item["master_port"],
                    "rank_offset": 1,
                    "world_size": 2,
                    "group_name": group_names[trainer_rank],
                    "backend": self.process_group_backend,
                }
            )

        await asyncio.gather(
            *[
                asyncio.to_thread(
                    _post_json,
                    url,
                    "init_weights_update_group",
                    payload,
                    self.request_timeout,
                )
                for url, payload in zip(self._server_urls, init_payloads, strict=True)
            ]
        )
        ray.get(init_refs)
        self._group_initialized = True
        print(
            "SGLang direct HCCL groups initialized: "
            f"groups={trainer_world_size}, world_size_per_group=2, backend={self.process_group_backend}",
            flush=True,
        )

    @auto_await
    async def update_weights(self, global_steps: int | None = None) -> None:
        await self.abort_replicas()
        await self._initialize_process_group()

        await self._post_all("resume_memory_occupation", {"tags": ["weights"]})
        ray.get(self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend))
        await self._post_all("resume_memory_occupation", {"tags": ["kv_cache"]})
        await self.resume_generation_replicas()
