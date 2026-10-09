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

"""Map Monarch failures onto Runtime RPC error types."""

from __future__ import annotations

import errno
from collections.abc import Iterator

from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
from monarch.actor import ActorError, MeshFailure

from verl.single_controller.base.errors import (
    PlacementUnavailableError,
    RPCRemoteError,
    RPCTransportError,
    RPCUnavailableError,
)


class MonarchAddressInUseError(OSError):
    """Report that Monarch failed to bind a worker rendezvous port."""

    def __init__(self, message: str = "Monarch rendezvous address already in use") -> None:
        super().__init__(errno.EADDRINUSE, message)


def _unwrap_actor_error(exc: BaseException) -> BaseException | None:
    if not isinstance(exc, ActorError):
        return None
    inner = getattr(exc, "exception", None)
    return inner if isinstance(inner, BaseException) else None


def _with_cause(error: BaseException, cause: BaseException) -> BaseException:
    error.__cause__ = cause
    return error


def _walk_causes(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current
        cause = current.__cause__
        context = current.__context__
        if isinstance(cause, BaseException):
            stack.append(cause)
        if isinstance(context, BaseException) and context is not cause:
            stack.append(context)


def is_address_in_use(exc: BaseException) -> bool:
    """Return True when ``exc`` is a real bind collision, not a string match."""
    for current in _walk_causes(exc):
        if isinstance(current, MonarchAddressInUseError):
            return True
        if isinstance(current, OSError) and current.errno == errno.EADDRINUSE:
            return True
    return False


def _is_unavailable(exc: BaseException) -> bool:
    return isinstance(exc, SupervisionError | RPCUnavailableError | MeshFailure) or type(exc).__name__ == "MeshFailure"


def map_monarch_exception(exc: BaseException) -> BaseException:
    """Translate a Monarch failure into the common RPC error hierarchy.

    Reconstructable application exceptions are returned as-is when carried by
    ActorError. Confirmed unavailability becomes RPCUnavailableError; unknown
    transport outcomes become RPCTransportError; otherwise RPCRemoteError.
    """
    # Keep classified failures and local observation timeouts unchanged.
    if isinstance(
        exc,
        TimeoutError
        | RPCTransportError
        | RPCUnavailableError
        | RPCRemoteError
        | PlacementUnavailableError
        | MonarchAddressInUseError,
    ):
        return exc
    if is_address_in_use(exc):
        return _with_cause(MonarchAddressInUseError(), exc)

    if _is_unavailable(exc):
        return _with_cause(RPCUnavailableError(str(exc) or type(exc).__name__), exc)

    if isinstance(exc, ConnectionError):
        return _with_cause(RPCTransportError(str(exc) or type(exc).__name__), exc)

    inner = _unwrap_actor_error(exc)
    if inner is not None:
        return map_monarch_exception(inner)

    if isinstance(exc, ActorError):
        return _with_cause(RPCRemoteError(str(exc) or type(exc).__name__), exc)
    if type(exc).__module__.startswith("monarch"):
        return _with_cause(RPCTransportError(str(exc) or type(exc).__name__), exc)
    return exc
