# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import os
import socket
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

import torch.distributed.rpc as rpc

from verl.single_controller.base import ClassWithInitArgs, ResourcePool, Worker, WorkerGroup
from verl.single_controller.torchrpc.node import NodeResource, node_manager
from verl.single_controller.torchrpc.utils import call_remote_actor, rref_to_here


def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    def func(*args, **kwargs):
        args, kwargs = dispatch_fn(self, *args, **kwargs)
        output = execute_fn(method_name, *args, **kwargs)
        if blocking:
            output = rref_to_here(output)
        output = collect_fn(self, output)
        return output

    return func


def _get_available_master_addr_port():
    if "TORCHRPC_ADDR" in os.environ:
        addr = os.environ["TORCHRPC_ADDR"]
        with socket.socket() as sock:
            sock.bind(("", 0))
            return addr, str(sock.getsockname()[1])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    addr_info = socket.getaddrinfo(
        master_addr,
        master_port,
        family=socket.AF_UNSPEC,  # let socket choose the address family
        type=socket.SOCK_DGRAM,  # use UDP to avoid actually create connection
    )
    for family, _, _, _, sockaddr in addr_info:
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                s.connect(sockaddr[:2])
                sockname = s.getsockname()
                return sockname[0], str(sockname[1])
        except OSError:
            continue
    raise Exception("No route to MASTER thus verl doesn't know which interface to use. Please set TORCHRPC_ADDR manually on each node.")


class TorchRPCResourcePool(ResourcePool):
    def __init__(
        self,
        process_on_nodes: List[int] = None,
        use_gpu: bool = True,
        max_colocate_count: int = 5,
    ) -> None:
        super().__init__(process_on_nodes, max_colocate_count)
        self.use_gpu = use_gpu
        self.resources = None

    def get_resources(self):
        if self.resources is not None:
            return self.resources

        self.resources = []
        for process_count in self._store:
            self.resources.append(node_manager.dispatch(process_count if self.use_gpu else 0))
        return self.resources


class TorchRPCClassWithInitArgs(ClassWithInitArgs):
    def __init__(self, cls, *args, **kwargs) -> None:
        super().__init__(cls, *args, **kwargs)
        self.env_vars = {}

    def update_env_vars(self, env_vars: Dict):
        self.env_vars.update(env_vars)

    def __call__(self, resource: NodeResource, gpus: List[int]) -> Any:
        return resource.create_actor(self.cls, self.args, self.kwargs, self.env_vars, gpus)


class TorchRPCWorkerGroup(WorkerGroup):
    def __init__(self, resource_pool: TorchRPCResourcePool = None, cls_with_init: TorchRPCClassWithInitArgs = None, **kwargs) -> None:
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.cls_with_init = cls_with_init
        # Whether the WorkerGroup is a Colocate WorkerGroup created by FusedWorker.
        self.fused_worker_used = cls_with_init.fused_worker_used
        # if a WorkerGroup is spawned from Colocate WorkerGroup, this indicates which sub-class is binded to this WorkerGroup.
        self.sub_cls_name = ""
        self._attrs_with_rrefs = ["_workers", "resource_pool"]

        self._init_with_resource_pool(resource_pool=resource_pool, cls_with_init=cls_with_init)

        if cls_with_init is not None:
            self._bind_worker_method(self.cls_with_init.cls, func_generator)

        self.wg_dict = None
        self.method_names = []

    def _is_worker_alive(self, worker):
        return True  # TODO

    def _init_with_resource_pool(self, resource_pool, cls_with_init):
        use_gpu = resource_pool.use_gpu

        resources = resource_pool.get_resources()
        self._world_size = resource_pool.world_size

        rank = -1
        for idx, resource in enumerate(resources):
            local_world_size = resource_pool.store[idx]
            for local_rank in range(local_world_size):
                rank += 1

                # we pass in environment variable at option so that Worker can use environment variable to set
                env_vars = {"WORLD_SIZE": str(self._world_size), "RANK": str(rank), "WG_BACKEND": "torchrpc", "LOCAL_WORLD_SIZE": str(local_world_size), "LOCAL_RANK": str(local_rank)}

                if rank == 0:
                    self._master_addr, self._master_port = rpc.remote(resource.node.name, _get_available_master_addr_port).to_here()

                env_vars["MASTER_ADDR"] = self._master_addr
                env_vars["MASTER_PORT"] = self._master_port

                cls_with_init.update_env_vars(env_vars)
                worker = cls_with_init(resource, [resource.gpus[local_rank]] if use_gpu else [])
                self._workers.append(worker)

    def __deepcopy__(self, memo):
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj

        for key, value in self.__dict__.items():
            if key in self._attrs_with_rrefs:
                new_obj.__dict__[key] = value
            else:
                new_obj.__dict__[key] = deepcopy(value, memo)

        return new_obj

    def spawn(self, prefix_set):
        wg_dict = dict()
        for key in prefix_set:
            new_wg = deepcopy(self)
            new_wg._bind_worker_method(self.cls_with_init.cls.raw_cls_dict[key], func_generator)
            new_wg.sub_cls_name = key
            wg_dict[key] = new_wg
        return wg_dict

    def fuse(self, prefix_set):
        if self.wg_dict is None:
            self.wg_dict = self.spawn(prefix_set)
        for role_name, role_wg in self.wg_dict.items():
            setattr(self, role_name, role_wg)
        self.method_names = self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

    def _execute_remote_single_worker(self, worker, method_name: str, *args, **kwargs):
        if self.fused_worker_used and method_name not in self.method_names:
            return call_remote_actor(worker, self.fused_worker_execute_fn_name, (f"{self.sub_cls_name}_fwmn_{method_name}",) + args, kwargs)
        # fused worker not used
        return call_remote_actor(worker, method_name, args, kwargs)

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        return self._execute_remote_single_worker(self._workers[0], method_name, *args, **kwargs)

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        return self.execute_rank_zero_async(method_name, *args, **kwargs).to_here()

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all_async(self, method_name: str, *args, **kwargs):
        length = len(self._workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(self._execute_remote_single_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs))
                return result

        return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs) for worker in self._workers]

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        ret = self.execute_all_async(method_name, *args, **kwargs)
        return rref_to_here(ret)

    def execute_all(self, method_name: str, *args, **kwargs):
        return self.execute_all_async(method_name, *args, **kwargs)


def torchrpc_remote(torchrpc_func):
    def wrapper():
        rank = int(os.environ.get("TORCHRPC_RANK"))
        world_size = int(os.environ.get("TORCHRPC_WORLD_SIZE"))
        rpc.init_rpc(f"torchrpc_worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(_transports=["uv"]))
        try:
            if rank == 0:
                node_manager.init()
                torchrpc_func()
        finally:
            rpc.shutdown()

    return wrapper


FusedWorkerCLSName = "FusedWorker"


def create_colocated_worker_raw_cls(class_dict: dict[str, TorchRPCClassWithInitArgs]):
    """
    This function returns a FusedWorker class.

    `FusedWorker.{class_name}` -> FusedClass
        Use `class_name` as a param to directly access the underlying class.

    `FusedWorker._fuw_execute("{class_name}_fwmn_{method_name}", *args, **kwargs)`
        First param must be "{class_name}_fwmn_{method_name}" in order to access `method_name`
        of underlying class `{class_name}`.

    `FusedWorker.fused_worker_dict` -> {"class_name": FusedClass}
        Stores all underlying classes.

    `FusedClass.fused_worker_dict` -> {"class_name": FusedClass}
        The same as `FusedWorker.fused_worker_dict`, enables underlying class to access other
        underlying classes.
    """
    raw_cls_dict = {cls_name: cia.cls for cls_name, cia in class_dict.items()}
    init_args_dict = {cls_name: cia.args for cls_name, cia in class_dict.items()}
    init_kwargs_dict = {cls_name: cia.kwargs for cls_name, cia in class_dict.items()}
    cls_names = list(class_dict.keys())

    # FusedWorker_Actor_Critic
    class_name_renamed = "_".join([FusedWorkerCLSName] + cls_names)

    class FusedWorker(Worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cls_names = cls_names
            self.raw_cls_dict = raw_cls_dict
            self.init_args_dict = init_args_dict
            self.init_kwargs_dict = init_kwargs_dict

            for cls_name, udc, ud_args, ud_kwargs in zip(self.cls_names, self.raw_cls_dict.values(), self.init_args_dict.values(), self.init_kwargs_dict.values()):
                with patch.dict(os.environ, {"DISABLE_WORKER_INIT": "1"}):
                    udc._get_ray_actor_cls_name = lambda x, name_renamed=class_name_renamed: name_renamed
                    udc._get_ray_method_prefix = lambda x, name_prefixed=cls_name: f"{name_prefixed}_"
                    # cls_name = "actor", "critic", udc = ActorWorker, CriticWorker
                    self.fused_worker_dict[cls_name] = udc(*ud_args, **ud_kwargs)
                    setattr(self, cls_name, self.fused_worker_dict[cls_name])

            # injecting fused_worker to each sub worker so they can be aware of existence of each other
            for _, worker in self.fused_worker_dict.items():
                setattr(worker, Worker.fused_worker_attr_name, self.fused_worker_dict)

        def _fuw_execute(self, method_name: str, *args, **kwargs):
            # for fused_worker, method_name is in a form of "{cls_name}_fwmn_{method_name}"
            # where fwmn stands "fused worker method name"
            names = method_name.split("_fwmn_")
            cls_name = names[0]
            method_name = names[1]

            assert cls_name in self.fused_worker_dict, f"calling {cls_name}'s {method_name}, but {cls_name} not in fused_worker_dict"
            udc_method = getattr(self.fused_worker_dict[cls_name], method_name)
            return udc_method(*args, **kwargs)

    renamed_fused_worker_cls = type(class_name_renamed, (FusedWorker,), {})
    renamed_fused_worker_cls.is_fused_worker = True
    renamed_fused_worker_cls.raw_cls_dict = raw_cls_dict

    return renamed_fused_worker_cls


def create_colocated_worker_cls(class_dict: dict[str, TorchRPCClassWithInitArgs]):
    """
    This function returns a RayClassWithInitArgs instance of FusedWorker, which is an replacement
    of `create_colocated_worker_cls`. WorkerGroup constructed using this class will be a colocated
    WorkerGroup, which will be referenced as `ColocateWorkerGroup` below.

    `ColocateWorkerGroup.spawn(prefix_set)`
        returns a dict of WorkerGroup {"class_name": WorkerGroup}, WorkerGroup in this dict will
        have methods of underlying class `class_name` attached.

    `ColocateWorkerGroup.fuse(prefix_set)`
        After executing this function, `ColocateWorkerGroup.{class_name}` will return WorkerGroup
        with methods of underlying class `class_name` attached.
    """
    raw_colocated_worker_cls = create_colocated_worker_raw_cls(class_dict)

    remote_cls = raw_colocated_worker_cls
    cia = TorchRPCClassWithInitArgs(cls=remote_cls)
    cia.fused_worker_used = True

    return cia
