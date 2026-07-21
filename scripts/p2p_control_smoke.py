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

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class P2PControlPlaneReplica(Protocol):
    async def abort_all_requests(self) -> None: ...

    async def get_remote_instance_transfer_engine_info(self, rank: int) -> Any: ...

    async def get_parallelism_info(self, rank: int) -> Any: ...

    async def begin_weight_update(self, selector: str = "all") -> dict[str, Any]: ...

    async def update_weight_version(self, weight_version: str, abort_all_requests: bool = True) -> dict[str, Any]: ...

    async def end_weight_update(self) -> dict[str, Any]: ...

    async def resume_generation(self) -> None: ...


def _assert_success(result: dict[str, Any], step: str) -> None:
    if result.get("success") is False:
        raise RuntimeError(f"P2P control-plane smoke failed at {step}: {result.get('message')}")


async def run_p2p_control_plane_smoke(
    replica: P2PControlPlaneReplica,
    *,
    num_ranks: int,
    weight_version: str = "smoke-1",
    selector: str = "all",
) -> dict[str, Any]:
    """Exercise Verl -> SGLang P2P control API without transferring weights.

    Lifecycle (Miles P2P order):
      abort -> metadata queries -> begin -> version bump -> end -> continue
    """
    if num_ranks <= 0:
        raise ValueError(f"num_ranks must be positive, got {num_ranks}")

    logger.info("P2P control-plane smoke: abort_all_requests")
    await replica.abort_all_requests()

    metadata: dict[int, dict[str, Any]] = {}
    for rank in range(num_ranks):
        logger.info("P2P control-plane smoke: query metadata for rank=%s", rank)
        transfer_info = await replica.get_remote_instance_transfer_engine_info(rank)
        parallelism_info = await replica.get_parallelism_info(rank)
        if transfer_info is None:
            raise RuntimeError(f"P2P control-plane smoke: missing transfer engine info for rank {rank}")
        if parallelism_info is None:
            raise RuntimeError(f"P2P control-plane smoke: missing parallelism config for rank {rank}")
        metadata[rank] = {
            "transfer_info": transfer_info,
            "parallelism_info": parallelism_info,
        }

    logger.info("P2P control-plane smoke: begin_weight_update selector=%s", selector)
    begin_result = await replica.begin_weight_update(selector)
    _assert_success(begin_result, "begin_weight_update")

    logger.info("P2P control-plane smoke: update_weight_version -> %s", weight_version)
    version_result = await replica.update_weight_version(weight_version, abort_all_requests=False)
    _assert_success(version_result, "update_weight_version")

    logger.info("P2P control-plane smoke: end_weight_update")
    end_result = await replica.end_weight_update()
    _assert_success(end_result, "end_weight_update")

    logger.info("P2P control-plane smoke: resume_generation")
    await replica.resume_generation()

    return {
        "metadata": metadata,
        "begin": begin_result,
        "update_weight_version": version_result,
        "end": end_result,
        "weight_version": weight_version,
    }
