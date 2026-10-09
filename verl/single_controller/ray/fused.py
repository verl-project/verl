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

"""Fused / colocated Ray worker class construction."""

from __future__ import annotations

import inspect

import ray

from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch
from verl.single_controller.base.fused import FusedWorker as BaseFusedWorker
from verl.single_controller.base.remote_worker_group import registered_methods
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.actor import RayClassWithInitArgs
from verl.utils.py_functional import temp_env_var

FusedWorkerCLSName = "FusedWorker"


def _make_role_forwarder(role_key: str, method_name: str, *, is_async: bool):
    def func(self, *args, **kwargs):
        return getattr(self.worker_dict[role_key], method_name)(*args, **kwargs)

    async def async_func(self, *args, **kwargs):
        return await getattr(self.worker_dict[role_key], method_name)(*args, **kwargs)

    return async_func if is_async else func


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        return cls.__ray_actor_class__
    return cls


def _determine_base_class(mros: list) -> type:
    for cls in mros[0]:
        if cls.__name__ == "MegatronWorker":
            return cls
        if cls.__name__ in ("Worker",):
            return cls
        # Prefer verl.single_controller.Worker / single_controller Worker by MRO name.
        if isinstance(cls, type) and issubclass(cls, Worker) and cls is not Worker:
            return cls
    for cls in mros[0]:
        if isinstance(cls, type) and issubclass(cls, Worker):
            return cls
    raise ValueError(f"Cannot determine base class for {mros}")


# deprecated, switching to FusedWorker
def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    """
    Binds the methods of each worker to the WorkerDict.
    Note that we only bind public methods that are decorated by register
    """
    backend_methods: dict[str, str] = {}
    for method_name, attrs in registered_methods(user_defined_cls).items():
        method = getattr(user_defined_cls, method_name)
        func = _make_role_forwarder(key, method_name, is_async=inspect.iscoroutinefunction(method))
        setattr(func, MAGIC_ATTR, attrs)
        if attrs["dispatch_mode"] == Dispatch.DIRECT_ROLLOUT_METHOD and "rollout" in key:
            if hasattr(cls, method_name):
                raise ValueError(f"conflict direct rollout method {method_name} with role {key}")
            backend_method = method_name
        else:
            backend_method = key + "_" + method_name
        setattr(cls, backend_method, func)
        backend_methods[method_name] = backend_method
    return backend_methods


# deprecated, switching to FusedWorker
def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = _determine_base_class([cls.cls.__ray_actor_class__.__mro__ for cls in class_dict.values()])
    for key, cls in class_dict.items():
        cls_dict[key] = cls.cls
        init_args_dict[key] = {"args": cls.args, "kwargs": cls.kwargs}

    class WorkerDict(worker_cls):
        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    worker = user_defined_cls(
                        *init_args_dict[key].get("args", ()),
                        **init_args_dict[key].get("kwargs", {}),
                    )
                worker._configure_with_store(self.__dict__)
                self.worker_dict[key] = worker
            for worker in self.worker_dict.values():
                setattr(worker, Worker.fused_worker_attr_name, self.worker_dict)

    role_method_metadata = {}
    role_backend_methods = {}
    for key, user_defined_cls in class_dict.items():
        raw_cls = _unwrap_ray_remote(user_defined_cls.cls)
        role_method_metadata[key] = registered_methods(raw_cls)
        role_backend_methods[key] = _bind_workers_method_to_parent(WorkerDict, key, raw_cls)

    remote_cls = ray.remote(WorkerDict)
    deferred = RayClassWithInitArgs(cls=remote_cls)
    deferred._colocated_role_method_metadata = role_method_metadata
    deferred._colocated_role_backend_methods = role_backend_methods
    return deferred


def create_colocated_worker_raw_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """Build an undecorated ``FusedWorker`` class that hosts every role in one process.

    Args:
        class_dict: Role name to deferred worker construction. Classes may be
            ``ray.remote`` wrappers; the raw Worker type is used for each role.

    Returns:
        A ``FusedWorker`` subclass named after the fused roles, not yet wrapped
        with ``ray.remote``.
    """
    raw_cls_dict = {cls_name: _unwrap_ray_remote(cia.cls) for cls_name, cia in class_dict.items()}
    init_args_dict = {cls_name: cia.args for cls_name, cia in class_dict.items()}
    init_kwargs_dict = {cls_name: cia.kwargs for cls_name, cia in class_dict.items()}
    cls_names = list(class_dict.keys())
    class_name_renamed = "_".join([FusedWorkerCLSName] + cls_names)

    role_manifests = {
        cls_name: {
            "module": user_cls.__module__,
            # The module exports the @ray.remote wrapper under the user's
            # class name. Resolve through that wrapper to the raw Worker type.
            "qualname": f"{user_cls.__qualname__}.__ray_actor_class__",
            "args": init_args_dict[cls_name],
            "kwargs": init_kwargs_dict[cls_name],
        }
        for cls_name, user_cls in raw_cls_dict.items()
    }

    class FusedWorker(BaseFusedWorker):
        fused_worker_attr_name = "fused_worker_dict"

        def __init__(self, *args, **kwargs):
            super().__init__(role_manifests)
            self.raw_cls_dict = raw_cls_dict
            self.init_args_dict = init_args_dict
            self.init_kwargs_dict = init_kwargs_dict
            for cls_name, worker in self.fused_worker_dict.items():
                worker._get_ray_actor_cls_name = lambda name_renamed=class_name_renamed: name_renamed
                worker._get_ray_method_prefix = lambda name_prefixed=cls_name: f"{name_prefixed}_"

    renamed = type(class_name_renamed, (FusedWorker,), {})
    renamed.is_fused_worker = True
    renamed.raw_cls_dict = raw_cls_dict
    return renamed


def create_colocated_worker_cls_fused(class_dict: dict[str, RayClassWithInitArgs]):
    """Return a Ray deferred construction for a ``FusedWorker`` hosting every role.

    Args:
        class_dict: Role name to deferred worker construction.

    Returns:
        A :class:`RayClassWithInitArgs` wrapping the ``ray.remote`` fused class,
        marked with ``fused_worker_used=True``.
    """
    raw_cls = create_colocated_worker_raw_cls(class_dict)
    remote_cls = ray.remote(raw_cls)
    cia = RayClassWithInitArgs(cls=remote_cls)
    cia.fused_worker_used = True
    return cia
