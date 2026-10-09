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

"""Deferred actor construction descriptors."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeAlias, TypeVar, cast

from verl.runtime.executor import RuntimeThreadPoolExecutor, run_in_thread
from verl.single_controller.base.worker import Worker

if TYPE_CHECKING:
    from verl.single_controller.base.fused import FusedWorker

ActorT = TypeVar("ActorT")


class ClassWithInitArgs(Generic[ActorT]):
    """Wrapper that stores constructor arguments for deferred instantiation.

    This class is particularly useful for remote class instantiation where
    the actual construction needs to happen at a different time or location.
    """

    def __init__(self, cls: type[ActorT], *args: Any, **kwargs: Any) -> None:
        """Initialize the ClassWithInitArgs instance.

        Args:
            cls: The class to be instantiated later.
            *args: Positional arguments for the class constructor.
            **kwargs: Keyword arguments for the class constructor.
        """
        if not isinstance(cls, type):
            raise TypeError(f"cls must be a type, got {type(cls)!r}")
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    @property
    def actor_class(self) -> type[ActorT]:
        """Return the stored actor class (alias of ``cls``)."""
        return self.cls

    @classmethod
    def check_actor_class(cls, actor_class: type) -> None:
        """Validate that a class can be reconstructed from an import manifest."""
        module = getattr(actor_class, "__module__", None)
        qualname = getattr(actor_class, "__qualname__", None)
        if not isinstance(module, str) or not module:
            raise ValueError(f"actor_class {actor_class!r} lacks a stable __module__")
        if not isinstance(qualname, str) or not qualname or "<" in qualname:
            raise ValueError(f"actor_class {actor_class!r} lacks a stable __qualname__")

    def __call__(self) -> ActorT:
        """Instantiate the stored class with the stored arguments."""
        return self.cls(*self.args, **self.kwargs)


class WorkerContainer:
    """Own one Worker, its execution thread, calls, and retryable close."""

    def __init__(
        self,
        actor: ClassWithInitArgs[Worker],
        *,
        thread_name_prefix: str,
        owner_thread_setup: Callable[[], None] | None = None,
    ) -> None:
        self._thread_pool: RuntimeThreadPoolExecutor | None = None
        self._worker: Worker | None = None
        self._runs_on_actor_loop = False
        self._active_calls = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._closing = False
        self._closed = False

        def setup_owner_thread() -> None:
            if owner_thread_setup is not None:
                owner_thread_setup()

        try:
            self._construct(
                actor,
                thread_name_prefix=thread_name_prefix,
                setup_owner_thread=setup_owner_thread,
            )
        except BaseException:
            self._abort_initialization()
            raise

    @property
    def _owned_worker(self) -> Worker:
        if self._worker is None:
            raise RuntimeError("Worker construction did not complete")
        return self._worker

    @property
    def _fused_worker(self) -> FusedWorker | None:
        from verl.single_controller.base.fused import FusedWorker

        return cast(FusedWorker, self._worker) if isinstance(self._worker, FusedWorker) else None

    def _construct(
        self,
        actor: ClassWithInitArgs[Worker],
        *,
        thread_name_prefix: str,
        setup_owner_thread: Callable[[], None],
    ) -> None:
        from verl.single_controller.base.fused import FusedWorker, _worker_class_is_async

        if issubclass(actor.cls, FusedWorker):
            actor_kwargs = dict(actor.kwargs)
            actor_kwargs["defer_role_init"] = True
            setup_owner_thread()
            self._worker = actor.cls(*actor.args, **actor_kwargs)
            self._initialize_fused_roles(async_capable=True)
            fused = cast(FusedWorker, self._worker)
            if any(not fused._fuw_role_is_async(name) for name in fused.cls_names):
                self._thread_pool = RuntimeThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=thread_name_prefix,
                )

                def initialize_sync_roles() -> None:
                    setup_owner_thread()
                    self._initialize_fused_roles(async_capable=False)

                self._thread_pool.submit(initialize_sync_roles).result()
            return

        self._runs_on_actor_loop = _worker_class_is_async(actor.cls)
        if self._runs_on_actor_loop:
            setup_owner_thread()
            self._worker = actor()
            return

        self._thread_pool = RuntimeThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name_prefix,
        )

        def initialize_sync_worker() -> None:
            setup_owner_thread()
            self._worker = actor()

        self._thread_pool.submit(initialize_sync_worker).result()

    def _abort_initialization(self) -> None:
        fused = self._fused_worker
        if fused is not None:
            roles = [
                name for name in fused.cls_names if name in fused.fused_worker_dict and fused._fuw_role_is_async(name)
            ]
            self._rollback_fused_roles(roles)
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True, cancel_futures=True)
        from verl.runtime.core import close_attached_runtime

        try:
            close_attached_runtime()
        except Exception:
            pass

    def _initialize_fused_roles(self, *, async_capable: bool) -> None:
        fused = self._fused_worker
        if fused is None:
            raise RuntimeError("fused Worker construction did not complete")
        existing = set(fused.fused_worker_dict)
        try:
            fused._initialize_roles(async_capable=async_capable)
            for role_name in fused.cls_names:
                if role_name in existing or fused._fuw_role_is_async(role_name) != async_capable:
                    continue
                fused.fused_worker_dict[role_name]._setup_visible_devices()
        except BaseException:
            created = [name for name in fused.cls_names if name not in existing and name in fused.fused_worker_dict]
            self._rollback_fused_roles(created)
            raise

    def _rollback_fused_roles(self, role_names: list[str]) -> None:
        fused = self._fused_worker
        if fused is None:
            return
        for role_name in reversed(role_names):
            role = fused.fused_worker_dict.get(role_name)
            close = getattr(role, "close", None) if role is not None else None
            try:
                if callable(close):
                    self._resolve(close)
            except BaseException:
                pass
            finally:
                fused.fused_worker_dict.pop(role_name, None)
                if getattr(fused, role_name, None) is role:
                    delattr(fused, role_name)

    @staticmethod
    def _resolve(function: Callable[[], Any]) -> Any:
        result = function()
        return asyncio.run(result) if inspect.isawaitable(result) else result

    async def _call(self, function: Callable[[], Any], *, actor_loop: bool) -> Any:
        if actor_loop:
            result = function()
            return await result if inspect.isawaitable(result) else result
        if self._thread_pool is None:
            raise RuntimeError("synchronous Worker thread is unavailable")
        return await run_in_thread(self._thread_pool, lambda: self._resolve(function))

    async def _close_fused_roles(self) -> None:
        from verl.single_controller.base.errors import ExceptionGroup

        fused = self._fused_worker
        if fused is None:
            raise RuntimeError("Worker is not fused")
        errors: list[Exception] = []
        for role_name in reversed(fused.cls_names):
            role = fused.fused_worker_dict.get(role_name)
            close = getattr(role, "close", None) if role is not None else None
            if not callable(close):
                continue
            try:
                await self._call(close, actor_loop=fused._fuw_role_is_async(role_name))
            except Exception as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("runtime failures", errors)

    async def _close_worker(self) -> None:
        from verl.runtime.core import close_attached_runtime

        worker = self._owned_worker
        try:
            if self._fused_worker is not None:
                await self._close_fused_roles()
            else:
                await self._call(worker.close, actor_loop=self._runs_on_actor_loop)
        finally:
            # Attached backends may synchronously wait for RPC completion; keep
            # that wait off the actor loop that delivers those completions.
            await asyncio.to_thread(close_attached_runtime)

    async def _release_worker(self) -> None:
        def release() -> None:
            self._worker = None

        if self._thread_pool is None:
            await asyncio.to_thread(release)
        else:
            await run_in_thread(self._thread_pool, release)

    async def _shutdown(self) -> None:
        if self._closed:
            return
        self._closing = True
        await self._idle.wait()
        await self._close_worker()
        await self._release_worker()
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True, cancel_futures=True)
            self._thread_pool = None
        self._closed = True

    async def dispatch(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Invoke one Worker method and materialize its result before transport."""
        if method_name == "_shutdown":
            await self._shutdown()
            return None
        if self._closing:
            raise RuntimeError("Worker actor is closing")

        self._active_calls += 1
        self._idle.clear()
        try:
            return await self._dispatch_open(method_name, args, kwargs)
        finally:
            self._active_calls -= 1
            if self._active_calls == 0:
                self._idle.set()

    async def _dispatch_open(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        worker = self._owned_worker
        fused = self._fused_worker
        if method_name == "_fuw_execute" and fused is not None:
            role_name, role_method_name, *role_args = args
            if fused._fuw_role_is_async(role_name):
                return await fused._fuw_execute(
                    role_name,
                    role_method_name,
                    *role_args,
                    **kwargs,
                )
            return await self._call(
                partial(
                    fused._fuw_execute_sync,
                    role_name,
                    role_method_name,
                    *role_args,
                    **kwargs,
                ),
                actor_loop=False,
            )

        method = getattr(worker, method_name)
        return await self._call(
            partial(method, *args, **kwargs),
            actor_loop=self._runs_on_actor_loop,
        )


ActorSpec: TypeAlias = type[ActorT] | ClassWithInitArgs[ActorT]
ActorRoleMap: TypeAlias = Mapping[str, ActorSpec[ActorT]]
