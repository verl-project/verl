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

"""Backend-neutral data plane used by the V1 rollout pipeline."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from typing import Any

from omegaconf import DictConfig, OmegaConf
from packaging.version import InvalidVersion, Version

ROLLOUT_DATA_BACKEND_ENV = "VERL_ROLLOUT_DATA_BACKEND"
TRANSFER_QUEUE_BACKEND = "transfer_queue"
MOONCAKE_BACKEND = "mooncake"


def _to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


class TransferQueueBackend:
    """Thin adapter that keeps the existing TransferQueue data path unchanged."""

    @staticmethod
    def _module():
        try:
            import transfer_queue as tq
        except ImportError as exc:
            raise ImportError(
                "The transfer_queue rollout backend requires TransferQueue. Install the version documented by verl."
            ) from exc
        return tq

    @staticmethod
    def _to_ref(meta: Any):
        from verl.protocol import RolloutDataRef

        return RolloutDataRef(
            keys=list(meta.keys),
            tags=list(meta.tags),
            partition_id=meta.partition_id,
            fields=None if meta.fields is None else list(meta.fields),
            extra_info=dict(meta.extra_info or {}),
        )

    def start(self, config: Any = None) -> None:
        self._module().init(config)
        from verl.utils import transferqueue_utils

        transferqueue_utils.TQ_INITIALIZED = True

    def shutdown(self) -> None:
        try:
            self._module().close()
        finally:
            from verl.utils import transferqueue_utils

            transferqueue_utils.TQ_INITIALIZED = False

    async def put_batch_async(self, **kwargs):
        return self._to_ref(await self._module().async_kv_batch_put(**kwargs))

    def put_batch(self, **kwargs):
        return self._to_ref(self._module().kv_batch_put(**kwargs))

    def get_batch(self, **kwargs):
        return self._module().kv_batch_get(**kwargs)

    def list(self, partition_id: str | None = None):
        return self._module().kv_list() if partition_id is None else self._module().kv_list(partition_id)

    def delete(self, **kwargs):
        return self._module().kv_clear(**kwargs)

    @staticmethod
    def release_result(_result: Any) -> None:
        pass

    def supports_checkpoint(self) -> bool:
        module = self._module()
        try:
            version_supported = Version(getattr(module, "__version__", "")) >= Version("0.1.9")
        except InvalidVersion:
            return False
        return version_supported and all(
            callable(getattr(module, name, None)) for name in ("save_checkpoint", "load_checkpoint")
        )

    def save_checkpoint(self, path: str, **kwargs) -> None:
        self._module().save_checkpoint(path, **kwargs)

    def load_checkpoint(self, path: str) -> None:
        self._module().load_checkpoint(path)


_backend: Any = None
_backend_lock = threading.RLock()


def configure_runtime(config: Any) -> dict[str, Any]:
    """Publish backend selection before Ray starts so child processes inherit it."""
    with _backend_lock:
        if _backend is not None:
            raise RuntimeError("cannot reconfigure a running rollout data backend")
        backend_config = _to_dict(config)
        if backend_config.get("name") == MOONCAKE_BACKEND:
            options = dict(backend_config.get("config") or {})
            backend_config["config"] = options
            catalog = dict(options.get("catalog") or {})
            options["catalog"] = catalog
            catalog.setdefault("actor_name", f"verl_mooncake_catalog_{uuid.uuid4().hex}")
        os.environ[ROLLOUT_DATA_BACKEND_ENV] = json.dumps(backend_config)
        return backend_config


def _runtime_config() -> dict[str, Any]:
    raw = os.getenv(ROLLOUT_DATA_BACKEND_ENV)
    return json.loads(raw) if raw else {"name": TRANSFER_QUEUE_BACKEND}


def backend_name(config: Any = None) -> str:
    return _to_dict(_runtime_config() if config is None else config).get("name", TRANSFER_QUEUE_BACKEND)


def _new_backend(*, host_catalog: bool = False):
    config = _runtime_config()
    name = config.get("name", TRANSFER_QUEUE_BACKEND)
    if name == TRANSFER_QUEUE_BACKEND:
        return TransferQueueBackend()
    if name == MOONCAKE_BACKEND:
        from verl.utils.mooncake_rollout_backend import MooncakeRolloutDataBackend

        return MooncakeRolloutDataBackend(
            _to_dict(config.get("config")),
            host_catalog=host_catalog,
        )
    raise ValueError(f"unsupported rollout data backend: {name!r}")


def init(
    transfer_queue_config: Any = None,
    *,
    host_catalog: bool = False,
) -> None:
    global _backend
    with _backend_lock:
        if _backend is not None:
            if host_catalog and backend_name() == MOONCAKE_BACKEND and not getattr(_backend, "host_catalog", False):
                raise RuntimeError("Mooncake rollout backend is already running as a catalog client")
            return
        backend = _new_backend(host_catalog=host_catalog)
        try:
            backend.start(transfer_queue_config if isinstance(backend, TransferQueueBackend) else None)
        except Exception:
            with contextlib.suppress(Exception):
                backend.shutdown()
            raise
        _backend = backend


def _ready_backend():
    init()
    return _backend


def close() -> None:
    global _backend
    with _backend_lock:
        if _backend is None:
            return
        _backend.shutdown()
        _backend = None


async def async_batch_put(**kwargs):
    return await _ready_backend().put_batch_async(**kwargs)


def batch_put(**kwargs):
    return _ready_backend().put_batch(**kwargs)


def batch_get(**kwargs):
    return _ready_backend().get_batch(**kwargs)


def release_result(result: Any) -> None:
    _ready_backend().release_result(result)


def prepare_for_dispatch(ref: Any) -> Any:
    """Restore TransferQueue's native metadata path before worker dispatch."""
    if backend_name() != TRANSFER_QUEUE_BACKEND:
        return ref
    from transfer_queue import KVBatchMeta

    return KVBatchMeta(
        keys=ref.keys,
        tags=ref.tags,
        partition_id=ref.partition_id,
        fields=ref.fields,
        extra_info=ref.extra_info,
    )


def rows_to_fields(rows: list[dict[str, Any]]):
    """Build rollout fields using the selected backend's sequence layout."""
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    return list_of_dict_to_tensordict(rows, jagged_1d=backend_name() == MOONCAKE_BACKEND)


@contextlib.contextmanager
def materialized_batch(**kwargs):
    result = batch_get(**kwargs)
    try:
        yield result
    finally:
        release_result(result)


def list_entries(partition_id: str | None = None):
    return _ready_backend().list(partition_id)


def clear(**kwargs):
    return _ready_backend().delete(**kwargs)


def supports_checkpoint() -> bool:
    return _ready_backend().supports_checkpoint()


def save_checkpoint(path: str, **kwargs) -> None:
    _ready_backend().save_checkpoint(path, **kwargs)


def load_checkpoint(path: str) -> None:
    _ready_backend().load_checkpoint(path)
