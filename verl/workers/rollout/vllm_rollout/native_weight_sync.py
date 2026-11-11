# Copyright 2026 Individual Contributor
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
"""Ray control plane for direct trainer-to-vLLM full-weight synchronization."""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import ray
from ray.actor import ActorHandle

from verl.workers.rollout.replica import RolloutMode, RolloutReplica, get_rollout_replica_class


def get_native_rollout_endpoints(replicas: Sequence[RolloutReplica]) -> tuple[int, list[tuple[ActorHandle, int]]]:
    """Assign a disjoint range of HCCL ranks to each independent vLLM engine."""
    if not replicas:
        raise ValueError("Native HCCL requires at least one rollout replica")
    vllm_replica = get_rollout_replica_class("vllm")
    next_rank = 1  # The trainer sender owns rank 0.
    endpoints = []
    seen = set()
    for replica in replicas:
        if not isinstance(replica, vllm_replica) or replica.rollout_mode != RolloutMode.STANDALONE:
            raise ValueError("Native HCCL requires standalone vLLM rollout replicas")
        if replica.config.disaggregation.enabled:
            raise ValueError("Native HCCL does not support prefill/decode disaggregation")
        server = replica.server_handle
        if server is None or replica.world_size < 1 or len(replica.workers) != replica.world_size:
            raise ValueError(f"Replica {replica.replica_rank} has an incomplete rollout worker group")
        if server._actor_id in seen:
            raise ValueError(f"Duplicate vLLM server for replica {replica.replica_rank}")
        seen.add(server._actor_id)
        endpoints.append((server, next_rank))
        next_rank += replica.world_size
    return next_rank, endpoints


class RayWeightSyncClient:
    """Synchronous VLLMWeightSyncClient; each call must reach every replica before waiting."""

    def __init__(self, endpoints: Sequence[tuple[ActorHandle, int]]):
        self.endpoints = endpoints

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        ray.get(
            [
                server.collective_rpc.remote(
                    "init_weight_transfer_engine", kwargs={"init_info": dict(init_info, rank_offset=offset)}
                )
                for server, offset in self.endpoints
            ]
        )

    def start_weight_update(self) -> None:
        ray.get([server.collective_rpc.remote("start_weight_update") for server, _ in self.endpoints])

    def update_weights(self, update_info: dict[str, Any]) -> None:
        ray.get(
            [
                server.collective_rpc.remote("update_weights", kwargs={"update_info": update_info})
                for server, _ in self.endpoints
            ]
        )

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        ray.get([server.finish_native_weight_update.remote(weight_version) for server, _ in self.endpoints])


async def finish_native_rollout_update(replicas: Sequence[RolloutReplica], global_steps: int | None) -> None:
    """Discard old-generation caches before recording the installed training step."""
    await asyncio.gather(*(replica.clear_kv_cache() for replica in replicas))
    if global_steps is not None:
        await asyncio.gather(*(replica.server_handle.set_global_steps.remote(global_steps) for replica in replicas))


def configure_native_hccl(server, engine_kwargs: dict[str, Any]) -> None:
    """Select the registered vLLM-Ascend worker engine for full-weight reloads."""
    if server.rollout_mode != RolloutMode.STANDALONE or server.config.disaggregation.enabled:
        raise ValueError("Native HCCL requires standalone vLLM replicas without prefill/decode disaggregation")
    if server.model_config.lora_rank > 0 or server.model_config.lora.get("rank", 0) > 0:
        raise ValueError("Native HCCL requires full weights from a trainer without LoRA")
    if server.config.quantization.enabled or getattr(server.model_config.hf_config, "quantization_config", None):
        raise ValueError("Native HCCL requires unquantized rollout weights")
    mtp = server.config.mtp
    if (mtp and mtp.enable and mtp.enable_rollout) or engine_kwargs.get("speculative_config"):
        raise ValueError("Native HCCL updates only the main model; a speculative drafter needs its own weight update")

    weight_config = engine_kwargs.get("weight_transfer_config", {})
    if not isinstance(weight_config, Mapping) or weight_config.get("backend", "hccl") != "hccl":
        raise ValueError("checkpoint_engine.backend=vllm_native_hccl requires weight_transfer_config.backend=hccl")
    engine_kwargs["weight_transfer_config"] = {**weight_config, "backend": "hccl"}
