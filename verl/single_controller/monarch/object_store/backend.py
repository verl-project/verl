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

"""Root-owned TorchStore backend lifecycle for Monarch."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from verl.single_controller.monarch.object_store.store import _run

if TYPE_CHECKING:
    from monarch.actor import ProcMesh

    from verl.single_controller.monarch.object_store.ref_count import RefCountActor


class TorchStoreBackend:
    """Own one process-global TorchStore instance and its volume mesh."""

    def __init__(
        self,
        store_name: str,
        *,
        proc_mesh: ProcMesh | None = None,
        ref_count_actors: RefCountActor | None = None,
        timeout_s: float = 300.0,
        local_cache_bytes: int = 0,
    ) -> None:
        if not isinstance(store_name, str) or not store_name:
            raise ValueError(f"store_name must be a non-empty str, got {store_name!r}")
        if not isinstance(timeout_s, int | float) or isinstance(timeout_s, bool) or timeout_s <= 0:
            raise ValueError(f"timeout_s must be a positive number, got {timeout_s!r}")
        if isinstance(local_cache_bytes, bool) or not isinstance(local_cache_bytes, int) or local_cache_bytes < 0:
            raise ValueError("local_cache_bytes must be a nonnegative integer")
        self._local_cache_bytes = local_cache_bytes
        self._store_name = store_name
        self._timeout_s = float(timeout_s)
        self._proc_mesh = proc_mesh
        self._ref_count_actors = ref_count_actors
        self._store_closed = False

    @classmethod
    def start(
        cls,
        *,
        pool,
        store_name: str,
        timeout_s: float,
        strategy: str = "host",
        local_cache_bytes: int = 0,
    ) -> TorchStoreBackend:
        """Start one CPU volume ProcMesh on ``pool`` and initialize the store.

        Every WorkerGroup receives the returned ``client_config`` through its
        AttachSpec. This owner only creates and owns the volume mesh plus the
        global store.
        """
        backend = cls(store_name, timeout_s=timeout_s, local_cache_bytes=local_cache_bytes)
        if strategy not in ("host", "local_rank"):
            raise ValueError(f"unsupported TorchStore strategy {strategy!r}")
        timeout_s = backend._timeout_s
        from verl.single_controller.monarch.cluster import get_monarch_cluster
        from verl.single_controller.monarch.resource_pool import MonarchResourcePool

        if not isinstance(pool, MonarchResourcePool):
            raise TypeError(f"TorchStore deployment requires MonarchResourcePool, got {type(pool)!r}")
        volume_processes_per_node = 1 if strategy == "host" else pool.processes_per_node
        num_storage_volumes = pool.nnodes * volume_processes_per_node
        cluster = get_monarch_cluster()
        volume_mesh = cluster.spawn_proc_mesh(
            pool._monarch_host_mesh,
            processes_per_node=volume_processes_per_node,
            device_type="cpu",
            name=f"torchstore_{_mesh_name(store_name)}",
            env_vars={},
        )
        try:
            initialized = volume_mesh.initialized
            if initialized is not None:
                initialized.get(timeout=timeout_s)
            backend._ref_count_actors = _run(
                _async_initialize(
                    store_name=store_name,
                    num_storage_volumes=num_storage_volumes,
                    mesh=volume_mesh,
                    strategy=strategy,
                ),
                timeout_s=timeout_s,
            )
        except BaseException:
            with suppress(Exception):
                volume_mesh.stop().get(timeout=timeout_s)
            raise
        backend._proc_mesh = volume_mesh
        return backend

    @property
    def client_config(self) -> dict[str, object]:
        """Serializable configuration used by every process-local client."""
        result: dict[str, object] = {"store_name": self._store_name}
        if self._ref_count_actors is not None:
            result["ref_count_actors"] = self._ref_count_actors
        if self._local_cache_bytes:
            result["local_cache_bytes"] = self._local_cache_bytes
        return result

    @property
    def closed(self) -> bool:
        return self._store_closed and self._proc_mesh is None

    def close(self) -> None:
        """Shut down TorchStore, then stop its volume ProcMesh."""
        if self.closed:
            return
        if self._proc_mesh is None:
            raise RuntimeError("Only the TorchStoreBackend returned by start() owns lifecycle")
        if not self._store_closed:
            _run(_async_shutdown(self._store_name, self._ref_count_actors), timeout_s=self._timeout_s)
            self._store_closed = True
        self._proc_mesh.stop().get(timeout=self._timeout_s)
        self._proc_mesh = None


def _mesh_name(store_name: str) -> str:
    name = "".join(character if character.isalnum() else "_" for character in store_name)
    return name[:48] or "store"


async def _async_initialize(
    *,
    store_name: str,
    num_storage_volumes: int,
    mesh: ProcMesh,
    strategy: str,
) -> RefCountActor:
    import torchstore as ts
    from torchstore.strategy import HostStrategy, LocalRankStrategy

    from verl.single_controller.monarch.object_store.ref_count import RefCountActor

    if strategy == "host":
        placement_strategy = HostStrategy()
    elif strategy == "local_rank":
        placement_strategy = LocalRankStrategy()
    else:
        raise ValueError(f"unsupported TorchStore strategy {strategy!r}")

    await ts.initialize(
        num_storage_volumes=num_storage_volumes,
        strategy=placement_strategy,
        store_name=store_name,
        mesh=mesh,
    )
    # Same ProcMesh and coordinates as native StorageVolume actors: host gets
    # one owner per host; local_rank gets one per volume rank. No new processes.
    return mesh.spawn(f"{_mesh_name(store_name)}_ref_counts", RefCountActor, store_name)


async def _async_shutdown(store_name: str, owners: RefCountActor | None = None) -> None:
    import torchstore as ts

    from verl.single_controller.monarch.object_store.ref_count import close_ref_counts

    if owners is not None:
        await close_ref_counts(owners)
    await ts.shutdown(store_name=store_name)


__all__ = ["TorchStoreBackend"]
