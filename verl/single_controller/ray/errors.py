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

"""Map Ray failures onto Runtime RPC error types."""

from __future__ import annotations

from ray.exceptions import OwnerDiedError, RayActorError, RayTaskError, RpcError, WorkerCrashedError

from verl.single_controller.base.errors import RPCRemoteError, RPCTransportError, RPCUnavailableError


def _unwrap_task_error(exc: BaseException) -> BaseException | None:
    if not isinstance(exc, RayTaskError):
        return None
    cause = getattr(exc, "cause", None)
    return cause if isinstance(cause, BaseException) else None


def _with_cause(error: BaseException, cause: BaseException) -> BaseException:
    error.__cause__ = cause
    return error


def map_ray_exception(exc: BaseException) -> BaseException:
    """Translate a Ray failure into the common RPC error hierarchy.

    Reconstructable application exceptions are returned as-is when carried by
    RayTaskError. Confirmed unavailability becomes RPCUnavailableError; unknown
    transport outcomes become RPCTransportError; otherwise RPCRemoteError.
    """
    # Keep classified failures and local observation timeouts unchanged.
    if isinstance(exc, TimeoutError | RPCTransportError | RPCUnavailableError | RPCRemoteError):
        return exc

    if isinstance(exc, RayActorError | OwnerDiedError | WorkerCrashedError):
        return _with_cause(RPCUnavailableError(str(exc) or type(exc).__name__), exc)
    if isinstance(exc, ConnectionError | RpcError):
        return _with_cause(RPCTransportError(str(exc) or type(exc).__name__), exc)

    inner = _unwrap_task_error(exc)
    if inner is not None:
        # Prefer the original application exception identity when present.
        return inner

    if isinstance(exc, RayTaskError):
        return _with_cause(RPCRemoteError(str(exc) or type(exc).__name__), exc)
    return exc
