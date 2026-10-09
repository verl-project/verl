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

"""Backend-independent fused Worker construction and invocation."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Mapping
from contextlib import suppress
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from verl.single_controller.base.actor import ActorRoleMap, ClassWithInitArgs
from verl.single_controller.base.worker import Worker
from verl.utils.py_functional import temp_env_var

if TYPE_CHECKING:
    from verl.single_controller.base.worker_group import WorkerGroup

GroupT = TypeVar("GroupT", bound="WorkerGroup[Worker]")


def normalize_role_actors(roles: ActorRoleMap[Worker]) -> dict[str, ClassWithInitArgs[Worker]]:
    """Normalize a role mapping into backend-independent constructors."""
    normalized: dict[str, ClassWithInitArgs[Worker]] = {}
    for role_name, actor in roles.items():
        if not isinstance(role_name, str) or not role_name:
            raise TypeError(f"fused role name must be a non-empty str, got {role_name!r}")
        if isinstance(actor, ClassWithInitArgs):
            normalized[role_name] = actor
        elif isinstance(actor, type):
            normalized[role_name] = ClassWithInitArgs(actor)
        else:
            raise TypeError(
                f"actor must be a Worker class, ClassWithInitArgs, or role-name mapping; got {type(actor)!r}"
            )
    return normalized


def role_manifests(roles: Mapping[str, ClassWithInitArgs[Worker]]) -> dict[str, dict[str, Any]]:
    """Convert normalized constructors into transport-serializable manifests."""
    manifests: dict[str, dict[str, Any]] = {}
    for role_name, init in roles.items():
        ClassWithInitArgs.check_actor_class(init.cls)
        manifests[role_name] = {
            "module": init.cls.__module__,
            "qualname": init.cls.__qualname__,
            "args": init.args,
            "kwargs": dict(init.kwargs),
        }
    return manifests


def create_fused_worker_groups(
    roles: ActorRoleMap[Worker],
    *,
    spawn_root: Callable[[ClassWithInitArgs[Worker]], GroupT],
) -> tuple[dict[str, GroupT], GroupT]:
    """Create one fused root and non-owning role views with shared rollback."""
    if not roles:
        raise ValueError("fused actor mapping must be non-empty")
    role_inits = normalize_role_actors(roles)
    root = spawn_root(
        ClassWithInitArgs(
            FusedWorker,
            role_manifests=role_manifests(role_inits),
        )
    )
    try:
        views = root._role_views(role_inits)
        for view in views.values():
            view._owned = False
            view._parent_group = root
    except BaseException:
        with suppress(Exception):
            root.close()
        raise
    return views, root


def _load_class(module_name: str, qualname: str) -> type[Worker]:
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type) or not issubclass(obj, Worker):
        raise TypeError(f"{module_name}.{qualname} is not a Worker type")
    return obj


class FusedWorker(Worker):
    """Own multiple role Workers in one process, independent of the transport."""

    is_fused_worker = True

    def __init__(
        self,
        role_manifests: Mapping[str, Mapping[str, Any]],
        *,
        defer_role_init: bool = False,
    ) -> None:
        super().__init__()
        self.cls_names = list(role_manifests)
        # Decode transport metadata once. Each owner thread constructs its
        # roles later from these descriptors without interpreting manifests.
        self._role_inits = {
            name: ClassWithInitArgs(
                _load_class(str(manifest["module"]), str(manifest["qualname"])),
                *tuple(manifest.get("args") or ()),
                **dict(manifest.get("kwargs") or {}),
            )
            for name, manifest in role_manifests.items()
        }
        self._role_async_capabilities = MappingProxyType(
            {name: _worker_class_is_async(init.cls) for name, init in self._role_inits.items()}
        )
        self.fused_worker_dict: dict[str, Worker] = {}
        if not defer_role_init:
            self._initialize_roles(async_capable=None)

    def _initialize_roles(self, *, async_capable: bool | None) -> None:
        for role_name, init in self._role_inits.items():
            if role_name in self.fused_worker_dict:
                continue
            if async_capable is not None and self._fuw_role_is_async(role_name) != async_capable:
                continue
            with temp_env_var("DISABLE_WORKER_INIT", "1"):
                role = init()
            role._configure_with_store(self.__dict__)
            self.fused_worker_dict[role_name] = role
            setattr(self, role_name, role)
        # Inject the shared role map so siblings can look each other up.
        for worker in self.fused_worker_dict.values():
            setattr(worker, Worker.fused_worker_attr_name, self.fused_worker_dict)

    def _fuw_method(self, role_name: str, method_name: str):
        role = self.fused_worker_dict.get(role_name)
        if role is None:
            raise KeyError(f"unknown fused Worker role {role_name!r}")
        return getattr(role, method_name)

    def _fuw_role_is_async(self, role_name: str) -> bool:
        """Return the immutable owner-thread capability for one fused role."""
        try:
            return self._role_async_capabilities[role_name]
        except KeyError:
            raise KeyError(f"unknown fused Worker role {role_name!r}") from None

    def _fuw_execute_sync(self, role_name: str, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = self._fuw_method(role_name, method_name)
        if inspect.iscoroutinefunction(method):
            raise TypeError(f"fused Worker method {role_name}.{method_name} is async")
        result = method(*args, **kwargs)
        return asyncio.run(result) if inspect.isawaitable(result) else result

    async def _fuw_execute(self, role_name: str, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke one role method and materialize its result before transport."""
        method = self._fuw_method(role_name, method_name)
        # Role state and CUDA thread-local state belong to this actor thread.
        result = method(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    def close(self):
        from verl.single_controller.base.errors import ExceptionGroup

        async def _close_roles() -> None:
            errors: list[Exception] = []
            for role_name in reversed(self.cls_names):
                role = self.fused_worker_dict.get(role_name)
                close = getattr(role, "close", None) if role is not None else None
                if not callable(close):
                    continue
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    errors.append(exc)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise ExceptionGroup("runtime failures", errors)

        needs_async = any(
            inspect.iscoroutinefunction(getattr(role, "close", None))
            for role in (self.fused_worker_dict.get(name) for name in self.cls_names)
            if role is not None
        )
        if needs_async:
            return _close_roles()

        errors: list[Exception] = []
        for role_name in reversed(self.cls_names):
            role = self.fused_worker_dict.get(role_name)
            close = getattr(role, "close", None) if role is not None else None
            if callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        asyncio.run(result)
                except Exception as exc:
                    errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("runtime failures", errors)


def _worker_class_is_async(role_cls: type[Worker]) -> bool:
    """Whether user code on ``role_cls`` declares any coroutine method."""
    for cls in role_cls.__mro__:
        if cls is Worker:
            break
        if any(inspect.iscoroutinefunction(value) for value in cls.__dict__.values()):
            return True
    return False
