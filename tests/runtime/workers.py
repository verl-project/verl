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
"""Shared Worker actors for verl.single_controller behavior tests."""

from __future__ import annotations

import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tensordict import TensorDict

from verl.protocol import DataProto
from verl.runtime import (
    Dispatch,
    Execute,
    RemoteWorkerGroup,
    Worker,
    register,
)
from verl.utils.net_utils import get_local_ip_address


class EchoWorker(Worker):
    """Minimal worker used by most behavior tests."""

    def __init__(self, base: int = 0) -> None:
        super().__init__()
        self.base = base
        self.seen: list[Any] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def raw_add(self, y: int) -> int:
        return self.base + y + self.rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def ping(self, x: int) -> int:
        return self.base + x + self.rank

    def data_proto(self, value: int) -> DataProto:
        return DataProto(batch=TensorDict({"value": [value + self.rank]}, batch_size=[1]))

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, blocking=True)
    def add_all(self, x: int) -> int:
        return self.base + x + self.rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def ping_async(self, x: int) -> int:
        return self.base + x + self.rank

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO, blocking=True)
    def rank_zero_sum(self, x: int, y: int) -> int:
        return self.base + x + y

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sleep_then(self, seconds: float, value: int) -> int:
        time.sleep(seconds)
        return value + self.rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def raise_app_error(self, message: str) -> None:
        raise ValueError(message)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def raise_async_error(self, message: str) -> None:
        raise RuntimeError(message)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.RANK_ZERO, blocking=False)
    def produce_payload(self, size: int, marker: int) -> dict[str, Any]:
        return {"marker": marker, "payload": bytes(size), "rank": self.rank}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def consume_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "marker": data["marker"],
            "size": len(data["payload"]),
            "producer_rank": data["rank"],
            "consumer_rank": self.rank,
            "pid": os.getpid(),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def record(self, value: Any) -> Any:
        self.seen.append(value)
        return value

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def record_later(self, value: Any) -> Any:
        self.seen.append(value)
        return value

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def get_seen(self) -> list[Any]:
        return list(self.seen)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def get_identity(self) -> dict[str, int | str]:
        from verl.runtime import current_runtime

        runtime = current_runtime()
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "runtime_id": runtime.runtime_id,
            "runtime_backend": runtime.backend,
            "runtime_path": "/".join(runtime.node_path),
            "master_addr": self.get_master_addr_port()[0],
            "master_port": self.get_master_addr_port()[1],
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def get_local_ip(self) -> str:
        return get_local_ip_address()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def object_store_roundtrip(self, key: str, value: Any) -> Any:
        from verl.runtime import delete, get, put

        reference = put(key, value)
        try:
            return get(reference)
        finally:
            delete(reference)


class ConstructorThreadContextWorker(Worker):
    """Observe RuntimeContext from a thread started by the actor constructor."""

    def __init__(self) -> None:
        from verl.runtime import current_runtime

        def identity() -> tuple[str, str, str]:
            runtime = current_runtime()
            return runtime.runtime_id, runtime.backend, "/".join(runtime.node_path)

        with ThreadPoolExecutor(max_workers=1) as executor:
            self.constructor_context = executor.submit(identity).result()
        super().__init__()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def get_constructor_context(self) -> tuple[str, str, str]:
        return self.constructor_context


class SlowGateWorker(Worker):
    """Worker that can prove whether a method body executed after close races."""

    def __init__(self) -> None:
        super().__init__()
        self.executed = 0

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def mark_executed(self, delay: float = 0.2) -> int:
        time.sleep(delay)
        self.executed += 1
        return self.executed


class NestedRuntimeWorker(Worker):
    """Create one child WorkerGroup through the backend-attached Runtime node."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def create_child(self, value: int) -> dict[str, object]:
        return self._create_child(value)

    def _create_child(self, value: int) -> dict[str, object]:
        from verl.runtime import ClassWithInitArgs, current_runtime

        runtime = current_runtime()
        pool = runtime.create_resource_pool(nnodes=1, processes_per_node=1, device_type="cpu")
        child = runtime.create_worker_group(ClassWithInitArgs(EchoWorker), on=pool)
        identity = child.get_identity()[0]
        stored = child.object_store_roundtrip(
            f"tests/nested/{'/'.join(runtime.node_path)}",
            {"value": value},
        )[0]
        return {
            "value": child.ping(value)[0],
            "stored": stored,
            "runtime_id": str(identity["runtime_id"]),
            "parent_path": "/".join(runtime.node_path),
            "child_path": str(identity["runtime_path"]),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def get_executed(self) -> int:
        return self.executed


class OverridePingWorker(EchoWorker):
    """Override behavior while inheriting the base registered RPC contract."""

    def ping(self, x: int) -> int:
        return self.base + x + self.rank + 100


class CheckpointProbeWorker(Worker):
    """Record checkpoint-manager control flow without a transport backend."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, **kwargs: Any) -> dict[str, int]:
        if method == "init_process_group":
            assert kwargs["world_size"] == self.world_size * 2
            self.events.append(f"init_process_group:{kwargs['rank']}")
        else:
            self.events.append(method)
        return {"rank": self.rank}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def update_weights(self, global_steps: int, mode: str | None = None) -> dict[str, int]:
        _ = mode
        self.events.append("update_weights")
        return {"global_steps": global_steps, "rank": self.rank}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def get_events(self) -> list[str]:
        return list(self.events)


class ActorRole(Worker):
    def __init__(self) -> None:
        super().__init__()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def add(self, x: float) -> float:
        return x + self.rank


class CriticRole(Worker):
    def __init__(self, val: float = 10.0) -> None:
        super().__init__()
        self.val = val

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, blocking=True)
    def sub(self, value: float) -> float:
        return value - self.val


class ThreadAffineRole(Worker):
    def __init__(self) -> None:
        super().__init__()
        self.owner_thread = threading.get_ident()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def runs_on_owner_thread(self) -> bool:
        return threading.get_ident() == self.owner_thread


class VisibilityProbeWorker(Worker):
    """GPU-resource probe that does not require a local CUDA driver."""

    def _setup_visible_devices(self) -> None:
        return

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def visible_devices(self) -> str:
        from verl.utils.device import get_visible_devices_keyword

        return os.environ.get(get_visible_devices_keyword().upper(), "")


class EndpointConsumerWorker(Worker):
    """Receive a serialized WorkerGroup endpoint and invoke it from a worker."""

    def __init__(self) -> None:
        super().__init__()
        self.endpoint: RemoteWorkerGroup | None = None

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def set_endpoint(self, endpoint: RemoteWorkerGroup) -> None:
        self.endpoint = endpoint

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def call_endpoint(self, value: int) -> list[int]:
        if self.endpoint is None:
            raise RuntimeError("endpoint has not been injected")
        return self.endpoint.submit("ping", args=(value,))
