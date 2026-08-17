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

"""Mooncake implementation of the V1 rollout data backend."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import asdict
from itertools import groupby
from typing import Any

from verl.protocol import RolloutDataRef

logger = logging.getLogger(__name__)


class MooncakeRolloutDataBackend:
    """Store rollout fragments with Mooncake and index them by logical key."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        host_catalog: bool = False,
    ):
        self.config = config or {}
        self.host_catalog = host_catalog
        self.store = None
        self.transfer = None
        self.buffer_pool = None
        self.catalog = None
        self.catalog_transfer = None
        self._read_token_lock = threading.Lock()
        self._pending_read_tokens: set[int] = set()

    def _store_config(self) -> dict[str, Any]:
        explicit = self.config.get("store_init_kwargs") or self.config.get("store")
        if explicit:
            config = dict(explicit)
        else:
            from mooncake.mooncake_config import MooncakeConfig

            config = asdict(MooncakeConfig.load_from_env())
        if "master_server_address" in config:
            config.setdefault("master_server_addr", config.pop("master_server_address"))
        if "device_name" in config:
            config.setdefault("rdma_devices", config.pop("device_name"))
        return config

    def start(self, _config: Any = None) -> None:
        import ray
        from mooncake.buffer_pool import BufferPool
        from mooncake.dataproto_catalog import (
            DataProtoCatalog,
            DataProtoCatalogTransfer,
        )
        from mooncake.store import MooncakeDistributedStore
        from mooncake.structured_object_store import MooncakeBundleTransfer

        self.store = MooncakeDistributedStore()
        rc = self.store.setup(self._store_config())
        if rc != 0:
            raise RuntimeError(f"MooncakeDistributedStore setup failed with code {rc}")
        self.buffer_pool = BufferPool(self.store, **dict(self.config.get("buffer_pool_kwargs") or {}))
        self.transfer = MooncakeBundleTransfer(
            self.store,
            key_prefix=self.config.get("key_prefix", "verl-rollout"),
            default_chunk_bytes=int(self.config.get("default_chunk_bytes", 64 * 1024**2)),
            buffer_pool=self.buffer_pool,
        )

        catalog_config = dict(self.config.get("catalog") or {})
        actor_name = catalog_config["actor_name"]
        namespace = catalog_config.get("namespace", "verl")
        if self.host_catalog:
            self.catalog = (
                ray.remote(DataProtoCatalog)
                .options(
                    name=actor_name,
                    namespace=namespace,
                )
                .remote()
            )
        else:
            try:
                self.catalog = ray.get_actor(actor_name, namespace=namespace)
            except Exception as exc:
                raise RuntimeError(f"Mooncake DataProto catalog {namespace}/{actor_name} is not running") from exc
        self.catalog_transfer = DataProtoCatalogTransfer(self.transfer, self._catalog_call)

    def shutdown(self) -> None:
        catalog_transfer = self.catalog_transfer
        first_error = None

        def cleanup(name, callback, *args, **kwargs):
            nonlocal first_error
            try:
                callback(*args, **kwargs)
                return True
            except Exception as exc:
                logger.exception("Failed to %s", name)
                if first_error is None:
                    first_error = exc
                return False

        with self._read_token_lock:
            pending_tokens = tuple(self._pending_read_tokens)
        catalog_ready = True
        for token in pending_tokens:
            released = cleanup("release Mooncake catalog read pin", self._release_read, token)
            catalog_ready = released and catalog_ready
        if catalog_transfer is not None:
            catalog_ready &= cleanup("close Mooncake catalog transfer", catalog_transfer.close)

        drained = not self.host_catalog or self.catalog is None
        if catalog_ready and self.host_catalog and self.catalog is not None:
            drained = catalog_transfer is None or cleanup("drain Mooncake catalog", catalog_transfer.drain)

        if catalog_ready and drained:
            pool_closed = self.buffer_pool is None or cleanup(
                "close Mooncake buffer pool", self.buffer_pool.close
            )
            if pool_closed:
                self.buffer_pool = None
                if self.store is not None and cleanup("close Mooncake Store", self.store.close):
                    self.store = None

        catalog = self.catalog
        if drained and self.host_catalog and catalog is not None:
            import ray

            if cleanup(
                "stop drained Mooncake catalog actor",
                ray.kill,
                catalog,
                no_restart=True,
            ):
                self.catalog = None
        if first_error is not None:
            raise first_error
        self.catalog = self.catalog_transfer = None
        self.buffer_pool = self.store = self.transfer = None

    async def put_batch_async(self, **kwargs):
        work = asyncio.create_task(asyncio.to_thread(self.put_batch, **kwargs))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            while not work.done():
                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError:
                    continue
            if not work.cancelled() and (exc := work.exception()) is not None:
                logger.error(
                    "Mooncake PUT failed while cancellation was pending",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            raise

    @staticmethod
    def _put_args(
        kwargs: dict[str, Any],
    ) -> tuple[list[str], str, list[dict[str, Any]] | None, Any]:
        keys = list(kwargs.get("keys") or [])
        raw_tags = kwargs.get("tags")
        tags = None if raw_tags is None else [dict(tag or {}) for tag in raw_tags]
        if not keys:
            raise ValueError("Mooncake rollout put requires at least one key")
        if len(keys) != len(set(keys)):
            raise ValueError("Mooncake rollout put keys must be unique")
        if tags is not None and len(tags) != len(keys):
            raise ValueError("tags must have the same length as keys")
        fields = kwargs.get("fields")
        if fields is None and tags is None:
            raise ValueError("Mooncake rollout put requires fields or tags")
        return keys, kwargs.get("partition_id") or "default", tags, fields

    @staticmethod
    def _to_dataproto(fields: Any):
        from tensordict import TensorDict

        from verl.protocol import DataProto

        if isinstance(fields, TensorDict):
            if len(fields.keys()) == 0:
                raise ValueError("Mooncake rollout fields cannot be empty")
            fields = DataProto.from_tensordict(fields)
        if not isinstance(fields, DataProto):
            raise TypeError(f"Mooncake rollout fields must be TensorDict or DataProto, got {type(fields).__name__}")
        return fields

    def _catalog_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        call = getattr(self.catalog, method)
        remote = getattr(call, "remote", None)
        if remote is None:
            return call(*args, **kwargs)
        import ray

        return ray.get(remote(*args, **kwargs))

    def _release_read(self, token: int) -> None:
        for attempt in range(2):
            try:
                self.catalog_transfer.release_read(token)
                with self._read_token_lock:
                    self._pending_read_tokens.discard(token)
                return
            except Exception:
                if attempt:
                    with self._read_token_lock:
                        self._pending_read_tokens.add(token)
                    raise
                logger.exception("Retrying Mooncake catalog read-pin release")

    def release_result(self, result: Any) -> None:
        self.catalog_transfer.release_result(result)

    def _replicate_config(self) -> Any:
        settings = dict(self.config.get("replicate_config") or {})
        if not settings:
            return None
        if "group_ids" in settings:
            raise ValueError("group_ids require Mooncake structured-object group semantics")

        from mooncake.store import ReplicateConfig

        config = ReplicateConfig()
        for name, value in settings.items():
            if not hasattr(config, name):
                raise ValueError(f"unsupported Mooncake replicate config field: {name!r}")
            setattr(config, name, value)
        return config

    def put_batch(self, **kwargs) -> RolloutDataRef:
        keys, partition, tags, fields = self._put_args(kwargs)
        data = None
        put_options = {}
        if fields is not None:
            data = self._to_dataproto(fields)
            chunk_bytes = kwargs.get("chunk_bytes")
            if chunk_bytes is None:
                chunk_bytes = self.config.get("chunk_bytes")
            field_schemas = kwargs.get("field_schemas")
            if field_schemas is None:
                field_schemas = self.config.get("field_schemas")
            put_options = {
                "namespace": self.config.get("namespace", "verl"),
                "stage": str(kwargs.get("stage") or "rollout"),
                "chunk_bytes": chunk_bytes,
                "field_schemas": field_schemas,
                "config": self._replicate_config(),
            }
        result = self.catalog_transfer.put(
            data,
            partition=partition,
            keys=keys,
            tags=tags,
            **put_options,
        )
        return RolloutDataRef(
            keys=keys,
            tags=result["tags"],
            partition_id=partition,
            fields=result["fields"],
        )

    def get_batch(self, **kwargs):
        from tensordict import TensorDict

        from verl.protocol import DataProto
        from verl.utils.tensordict_utils import (
            assign_non_tensor_data,
            concat_tensordict,
            index_select_tensor_dict,
        )

        keys = list(kwargs.get("keys") or [])
        if not keys:
            return TensorDict({}, batch_size=[0])
        fields = kwargs.get("select_fields")
        if isinstance(fields, str):
            fields = [fields]
        catalog_transfer = self.catalog_transfer
        plan = catalog_transfer.resolve(
            kwargs.get("partition_id") or "default",
            keys,
            fields,
        )
        read_token = plan["read_token"]

        requests: dict[str, dict[str, Any]] = {}
        for field_group in plan["field_groups"]:
            for fragment_id, source_row in field_group["locations"]:
                request = requests.setdefault(fragment_id, {"rows": [], "row_indexes": {}, "fields": []})
                if source_row not in request["row_indexes"]:
                    request["row_indexes"][source_row] = len(request["rows"])
                    request["rows"].append(source_row)
                for field in field_group["fields"]:
                    if field not in request["fields"]:
                        request["fields"].append(field)

        materialized = {}
        materialized_rows = {}
        materialized_results = []
        try:
            for fragment_id, request in requests.items():
                rows = request["rows"]
                batch_size = int(plan["handles"][fragment_id]["batch_size"])
                if rows == list(range(batch_size)):
                    row_selection = None
                elif all(row == rows[0] + offset for offset, row in enumerate(rows)):
                    row_selection = slice(rows[0], rows[-1] + 1)
                else:
                    row_selection = rows
                data = self.transfer.get(
                    plan["handles"][fragment_id],
                    type="dataproto",
                    fields=request["fields"],
                    rows=row_selection,
                    data_cls=DataProto,
                )
                materialized_rows[fragment_id] = request["row_indexes"]
                materialized_results.append(data)
                materialized[fragment_id] = data.to_tensordict()

            result = TensorDict({}, batch_size=[len(keys)])
            for field_group in plan["field_groups"]:
                parts = []
                for fragment_id, group in groupby(field_group["locations"], key=lambda location: location[0]):
                    indexes = [materialized_rows[fragment_id][source_row] for _, source_row in group]
                    selected = materialized[fragment_id].select(*field_group["fields"])
                    if indexes != list(range(len(selected))):
                        selected = index_select_tensor_dict(selected, indexes)
                    parts.append(selected)
                merged = parts[0] if len(parts) == 1 else concat_tensordict(parts)
                result.update(merged)
            for name, value in plan["meta_info"].items():
                assign_non_tensor_data(result, name, value)
        except BaseException:
            catalog_transfer.discard_results(materialized_results)
            try:
                self._release_read(read_token)
            except Exception:
                logger.exception("Failed to release Mooncake catalog read pin")
            raise
        try:
            self._release_read(read_token)
        except Exception:
            catalog_transfer.discard_results(materialized_results)
            raise
        catalog_transfer.attach_results(result, materialized_results)
        return result

    def list(self, partition_id: str | None = None):
        return self.catalog_transfer.list(partition_id)

    def delete(self, **kwargs) -> None:
        keys = list(kwargs.get("keys") or [])
        if not keys:
            return
        self.catalog_transfer.remove(
            kwargs.get("partition_id") or "default",
            keys,
        )

    def supports_checkpoint(self) -> bool:
        return False
