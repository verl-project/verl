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
import multiprocessing as mp
import os
import threading
from typing import List, Union

import cloudpickle
import torch.distributed.rpc as rpc


def _local_actor_runner(pickled_args, input_queue, output_queue):
    """
    Main Loop of LocalActor

    This function runs in a seperate subprocess.
    It sets environment variables using env_vars, initializes `cls(*args, **kwargs)`, and then wait for commands from input_queue, executes the command, and puts the result to output_queue.

    Args:
        - pickled_args: a tuple of (cls, args, kwargs, env_vars) pickled by cloudpickle.
                        we use cloudpickle on top of default pickler to support tranmitting local object across processes such as colocated worker classes.
        - input_queue: input queue for command
                       command is a tuple of (method_name, args, kwargs) pickled by cloudpickle, which lets the actor run `cls.method_name(*args, **kwargs)`
        - output_queue: output queue for result
    """
    cls, args, kwargs, env_vars = cloudpickle.loads(pickled_args)
    if env_vars is not None:
        for k, v in env_vars.items():
            os.environ[k] = v
    inst = cls(*args, **kwargs)
    while True:
        pickled_cmd = input_queue.get()
        method_name, args, kwargs = cloudpickle.loads(pickled_cmd)
        method = getattr(inst, method_name)
        result = method(*args, **kwargs)
        output_queue.put(result)


class LocalActor:
    """
    An actor of cls that runs in a seperate subprocess.

    Args:
        - cls: class of the actor
        - args: arguments for initating the actor
        - kwargs: keyword arguments for initating the actor
        - env_vars: environment variables for the actor

    Methods:
        - `run(method_name, args, kwargs)`: run `cls.method_name(*args, **kwargs)` on the worker in the subprocess.
    """

    def __init__(self, cls, args, kwargs, env_vars):
        self.running_lock = threading.Lock()
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        # set daemon=True to make sure the subprocess is killed when the main process exits
        self.process = ctx.Process(target=_local_actor_runner, args=(cloudpickle.dumps((cls, args, kwargs, env_vars)), self.input_queue, self.output_queue), daemon=True)
        self.process.start()

    def run(self, method_name, args, kwargs):
        if method_name == "__del__":
            self.process.kill()
            return
        # only one job can be submitted at a time
        with self.running_lock:
            self.input_queue.put(cloudpickle.dumps((method_name, args, kwargs)))
            return self.output_queue.get()


class LocalActorManager:
    """
    Manage Local Actors on each Remote Node

    Only create one LocalActorManager per node.

    Methods:
        - `create_local_actor(cls, args, kwargs, env_vars, gpus)`: create a LocalActor of `cls` on the node.
        - `visible_gpus()`: return visible gpus on the node.
        - `run(method_name, args, kwargs)`: run `self.method_name(*args, **kwargs)`.
                                            this enables LocalActorManager to be called using the same way of LocalActor.
    """

    def __init__(self):
        self.lock = threading.Lock()

    def create_local_actor(self, cls, args, kwargs, env_vars, gpus) -> LocalActor:
        with self.lock:
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpus])
            actor = LocalActor(cls, args, kwargs, env_vars)
            if original_cuda_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        return actor

    def visible_gpus(self) -> List[int]:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is None:
            import torch

            return list(range(torch.cuda.device_count()))
        else:
            return [int(i) for i in cuda_visible_devices.split(",")]


def _call_local_actor(actor_rref: rpc.RRef, method_name, args, kwargs):
    """
    Run `actor.method_name(*args, **kwargs)` on the local actor.

    Don't call this function directly. Use `call_remote_actor` instead.
    """
    actor = actor_rref.local_value()
    if isinstance(actor, LocalActor):
        return actor.run(method_name, args, kwargs)
    else:
        return getattr(actor, method_name)(*args, **kwargs)


def call_remote_actor(actor_rref: rpc.RRef, method_name, args, kwargs):
    """
    Run `actor.method_name(*args, **kwargs)` on the remote actor.

    Args:
        - actor_rref: RRef of the actor
        - method_name: method name
        - args: arguments for the method
    """
    return rpc.remote(actor_rref.owner(), _call_local_actor, args=(actor_rref, method_name, args, kwargs))


def rref_to_here(x: Union[rpc.RRef, List[rpc.RRef]]):
    """
    Pull Remote Objects.

    `x` could be `rpc.RRef` or `List[rpc.RRef]`.
    """
    if isinstance(x, list):
        return [i.to_here() for i in x]
    else:
        return x.to_here()
