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

import asyncio
from typing import Any

import pytest

from scripts.p2p_control_smoke import run_p2p_control_plane_smoke


class _RecordingReplica:
    def __init__(self, num_ranks: int) -> None:
        self.num_ranks = num_ranks
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, *args, **kwargs) -> None:
        self.calls.append((name, args, kwargs))

    async def abort_all_requests(self) -> None:
        self._record("abort_all_requests")

    async def get_remote_instance_transfer_engine_info(self, rank: int) -> Any:
        self._record("get_remote_instance_transfer_engine_info", rank)
        return (f"session-{rank}", {f"weight.{rank}": (0, 1, 4)})

    async def get_parallelism_info(self, rank: int) -> dict[str, Any]:
        self._record("get_parallelism_info", rank)
        return {"tp_rank": rank, "tp_size": self.num_ranks}

    async def begin_weight_update(self, selector: str = "all") -> dict[str, Any]:
        self._record("begin_weight_update", selector)
        return {"success": True, "message": "ok"}

    async def update_weight_version(self, weight_version: str, abort_all_requests: bool = True) -> dict[str, Any]:
        self._record(
            "update_weight_version",
            weight_version,
            abort_all_requests=abort_all_requests,
        )
        return {"success": True, "message": "ok", "new_version": weight_version}

    async def end_weight_update(self) -> dict[str, Any]:
        self._record("end_weight_update")
        return {"success": True, "message": "ok"}

    async def resume_generation(self) -> None:
        self._record("resume_generation")


def test_p2p_control_plane_smoke_call_order():
    replica = _RecordingReplica(num_ranks=2)

    async def _run() -> None:
        result = await run_p2p_control_plane_smoke(replica, num_ranks=2, weight_version="42")
        assert result["weight_version"] == "42"
        assert set(result["metadata"]) == {0, 1}

    asyncio.run(_run())

    call_names = [name for name, _, _ in replica.calls]
    assert call_names == [
        "abort_all_requests",
        "get_remote_instance_transfer_engine_info",
        "get_parallelism_info",
        "get_remote_instance_transfer_engine_info",
        "get_parallelism_info",
        "begin_weight_update",
        "update_weight_version",
        "end_weight_update",
        "resume_generation",
    ]

    _, _, update_kwargs = replica.calls[6]
    assert update_kwargs["abort_all_requests"] is False


def test_p2p_control_plane_smoke_fails_on_missing_metadata():
    class _BrokenReplica(_RecordingReplica):
        async def get_parallelism_info(self, rank: int) -> dict[str, Any]:
            return None

    async def _run() -> None:
        replica = _BrokenReplica(num_ranks=1)
        await run_p2p_control_plane_smoke(replica, num_ranks=1)

    with pytest.raises(RuntimeError, match="missing parallelism config"):
        asyncio.run(_run())


def test_p2p_control_plane_smoke_sync_entrypoint():
    asyncio.run(run_p2p_control_plane_smoke(_RecordingReplica(1), num_ranks=1))
