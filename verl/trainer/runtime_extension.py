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

"""Versioned runtime extension points for trainer-side consumers."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

RUNTIME_EXTENSION_API_VERSION = 1


class RuntimeCapability(str, Enum):
    """Capabilities that a trainer may grant to an extension."""

    EVENTS = "events"
    SAFE_POINTS = "safe_points"


class SafePoint(str, Enum):
    """Stable trainer locations where an extension may coordinate work."""

    BEFORE_FIT = "before_fit"
    STEP_BOUNDARY = "step_boundary"
    WAITING_FOR_SAMPLES = "waiting_for_samples"
    AFTER_FIT = "after_fit"
    TERMINAL_FAILURE = "terminal_failure"


class SafePointDecision(str, Enum):
    """Decision returned by an extension at a safe point."""

    CONTINUE = "continue"
    RESTART_REQUIRED = "restart_required"


@dataclass(frozen=True)
class SampleWaitState:
    """Progress reported while a replay buffer waits for sampleable groups."""

    trainer_mode: str
    partition_id: str
    global_step: int
    required: int
    available: int
    pending: int
    running: int
    elapsed_seconds: float
    poll_interval: float
    poll_count: int


@dataclass(frozen=True)
class RuntimeEvent:
    """An immutable event published by the trainer or one of its adapters."""

    topic: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True)
class RuntimeExtensionContext:
    """Public context passed to runtime extensions."""

    run_id: str
    api_version: int
    capabilities: frozenset[RuntimeCapability]


@runtime_checkable
class RuntimeExtension(Protocol):
    """Protocol implemented by a runtime extension consumer."""

    name: str
    api_version: int
    required_capabilities: Iterable[RuntimeCapability]

    def on_event(self, context: RuntimeExtensionContext, event: RuntimeEvent) -> None:
        """Observe a runtime event."""

    def on_safe_point(self, context: RuntimeExtensionContext, safe_point: SafePoint) -> SafePointDecision | None:
        """Coordinate work at a safe execution point."""


class RuntimeExtensionRestartRequired(RuntimeError):
    """Raised when an extension cannot continue the current trainer process."""


class RuntimeEventBus:
    """Synchronous, ordered event delivery for registered consumers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[RuntimeEvent], None]]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, topic: str, callback: Callable[[RuntimeEvent], None]) -> Callable[[], None]:
        """Subscribe to a topic and return an idempotent unsubscribe callback."""
        with self._lock:
            self._subscribers[topic].append(callback)
        closed = False

        def unsubscribe() -> None:
            nonlocal closed
            with self._lock:
                if closed:
                    return
                closed = True
                self._subscribers[topic].remove(callback)
                if not self._subscribers[topic]:
                    del self._subscribers[topic]

        return unsubscribe

    def publish(self, event: RuntimeEvent) -> None:
        """Deliver an event to exact-topic and wildcard subscribers."""
        with self._lock:
            callbacks = tuple(self._subscribers.get(event.topic, ())) + tuple(self._subscribers.get("*", ()))
        for callback in callbacks:
            callback(event)


class RuntimeExtensionRegistration:
    """Closeable ownership handle for one registered extension."""

    def __init__(self, registry: RuntimeExtensionRegistry, name: str) -> None:
        self._registry = registry
        self.name = name
        self._closed = False

    def close(self) -> None:
        """Remove the extension and all subscriptions it owns."""
        if self._closed:
            return
        self._closed = True
        self._registry._unregister(self.name)

    def __enter__(self) -> RuntimeExtensionRegistration:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RuntimeExtensionRegistry:
    """Validate, own, and invoke runtime extensions for one trainer."""

    def __init__(
        self,
        *,
        capabilities: Iterable[RuntimeCapability],
        run_id: str | None = None,
    ) -> None:
        self.context = RuntimeExtensionContext(
            run_id=run_id or uuid.uuid4().hex,
            api_version=RUNTIME_EXTENSION_API_VERSION,
            capabilities=frozenset(capabilities),
        )
        self.events = RuntimeEventBus()
        self._extensions: dict[str, RuntimeExtension] = {}
        self._unsubscribe: dict[str, Callable[[], None]] = {}
        self._lock = threading.RLock()

    def __bool__(self) -> bool:
        """Return whether at least one extension is currently registered."""
        with self._lock:
            return bool(self._extensions)

    def register(self, extension: RuntimeExtension) -> RuntimeExtensionRegistration:
        """Register one consumer after validating its identity and capabilities."""
        name = extension.name.strip()
        if not name:
            raise ValueError("runtime extension name must be non-empty")
        if extension.api_version != self.context.api_version:
            raise ValueError(
                f"runtime extension {name!r} requires API version {extension.api_version}, "
                f"but the trainer provides {self.context.api_version}"
            )
        required = frozenset(extension.required_capabilities)
        missing = required - self.context.capabilities
        if missing:
            values = ", ".join(sorted(capability.value for capability in missing))
            raise ValueError(f"runtime extension {name!r} requires unavailable capabilities: {values}")

        with self._lock:
            if name in self._extensions:
                raise ValueError(f"runtime extension {name!r} is already registered")
            self._extensions[name] = extension
            if RuntimeCapability.EVENTS in required:
                self._unsubscribe[name] = self.events.subscribe(
                    "*", lambda event, consumer=extension: consumer.on_event(self.context, event)
                )
        return RuntimeExtensionRegistration(self, name)

    def reach(self, safe_point: SafePoint) -> None:
        """Invoke registered safe-point consumers in registration order."""
        with self._lock:
            extensions = tuple(self._extensions.values())
        for extension in extensions:
            if RuntimeCapability.SAFE_POINTS not in frozenset(extension.required_capabilities):
                continue
            decision = extension.on_safe_point(self.context, safe_point) or SafePointDecision.CONTINUE
            if decision is SafePointDecision.RESTART_REQUIRED:
                raise RuntimeExtensionRestartRequired(
                    f"runtime extension {extension.name!r} requested restart at {safe_point.value}"
                )
            if decision is not SafePointDecision.CONTINUE:
                raise ValueError(f"runtime extension {extension.name!r} returned invalid decision {decision!r}")

    def publish(self, topic: str, **payload: Any) -> RuntimeEvent:
        """Create and synchronously publish an event."""
        event = RuntimeEvent(topic=topic, payload=payload)
        self.events.publish(event)
        return event

    def close(self) -> None:
        """Close all registrations."""
        with self._lock:
            names = tuple(self._extensions)
        for name in names:
            self._unregister(name)

    def _unregister(self, name: str) -> None:
        with self._lock:
            unsubscribe = self._unsubscribe.pop(name, None)
            self._extensions.pop(name, None)
        if unsubscribe is not None:
            unsubscribe()
