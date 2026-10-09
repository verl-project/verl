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

"""Deferred Ray actor construction helpers."""

from __future__ import annotations

import os
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.plugin.platform import get_platform
from verl.single_controller.base.actor import ClassWithInitArgs, WorkerContainer
from verl.utils.device import get_resource_name, get_torch_device
from verl.utils.ray_utils import ray_noset_visible_devices


def _assign_ray_visible_devices() -> None:
    """Honor RAY_EXPERIMENTAL_NOSET_* by applying Ray's accelerator assignment."""
    if not ray_noset_visible_devices():
        return
    # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
    # environment variable for each actor, unless
    # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set,
    # so we need to set local rank when the flag is set.
    device_name = get_resource_name()
    accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
    assigned = accelerator_ids.get(device_name) or accelerator_ids.get(str(device_name).upper())
    if not assigned:
        return
    local_rank = str(assigned[0])
    os.environ["LOCAL_RANK"] = local_rank
    get_torch_device().set_device(int(local_rank))


def _runtime_bound_actor_class(actor_cls: type[Any]) -> type[Any]:
    """Build a Ray proxy with the same sync/async execution contract as Monarch."""

    class RayWorkerActor:
        def __init__(
            self,
            *args: Any,
            _verl_attach_spec=None,
            **kwargs: Any,
        ) -> None:
            if _verl_attach_spec is not None:
                from verl.runtime.core import Runtime

                Runtime._attach(_verl_attach_spec)
            self._inner = WorkerContainer(
                ClassWithInitArgs(actor_cls, *args, **kwargs),
                thread_name_prefix="verl-ray-worker",
                owner_thread_setup=_assign_ray_visible_devices,
            )

    def make_remote_method(method_name: str):
        async def invoke(self, *args: Any, **kwargs: Any) -> Any:
            return await self._inner.dispatch(method_name, args, kwargs)

        invoke.__name__ = method_name
        invoke.__qualname__ = f"RuntimeBound{actor_cls.__name__}.{method_name}"
        return invoke

    for method_name in dir(actor_cls):
        if method_name.startswith("__") or method_name == "close":
            continue
        try:
            method = getattr(actor_cls, method_name)
        except Exception:  # noqa: BLE001 - class descriptors may reject access
            continue
        if callable(method):
            if method_name == "_inner":
                raise ValueError("Worker method '_inner' is reserved by the Runtime actor")
            setattr(RayWorkerActor, method_name, make_remote_method(method_name))

    RayWorkerActor.__name__ = f"RuntimeBound{actor_cls.__name__}"
    return RayWorkerActor


class RayClassWithInitArgs:
    """A wrapper class for Ray actors with initialization arguments.

    This class provides additional functionality for configuring and creating
    Ray actors with specific resource requirements and scheduling strategies.
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self._runtime_user_cls: type[Any] | None = None
        self._options: dict[str, Any] = {}
        self.fused_worker_used = False

    @classmethod
    def from_class_init(cls, init: ClassWithInitArgs[Any]) -> RayClassWithInitArgs:
        remote_cls = ray.remote(_runtime_bound_actor_class(init.cls))
        deferred = cls(remote_cls, *init.args, **dict(init.kwargs))
        deferred._runtime_user_cls = init.cls
        return deferred

    def update_options(self, options: dict):
        """Update the Ray actor creation options.

        Args:
            options: Dictionary of options to update
        """
        self._options.update(options)

    def __call__(
        self,
        use_gpu: bool = True,
        num_gpus=1,
        sharing_with=None,
        device_name="cuda",
        scheduling_strategy=None,
        attach_spec=None,
    ) -> Any:
        actor_kwargs = dict(self.kwargs)
        if attach_spec is not None:
            actor_kwargs["_verl_attach_spec"] = attach_spec
        if sharing_with is not None:
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            return self.cls.options(**options).remote(
                *self.args,
                cuda_visible_devices=visible_devices,
                **actor_kwargs,
            )

        if scheduling_strategy is None:
            raise ValueError("Ray Runtime actor creation requires hard node affinity")
        options = {"scheduling_strategy": scheduling_strategy}
        options.update(self._options)

        if use_gpu:
            resource_opts = get_platform().ray_resource_options(num_gpus)
            options.update(resource_opts)

        return self.cls.options(**options).remote(*self.args, **actor_kwargs)
