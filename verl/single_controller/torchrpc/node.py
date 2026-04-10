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
from typing import List

import torch.distributed.rpc as rpc

from verl.single_controller.torchrpc.utils import LocalActorManager, call_remote_actor


class Node:
    """
    Dispatch GPU resources on a node

    Should be created by `NodeManager`.

    Args:
        - name: name of the node

    Members:
        - name: name of the node
        - local_actor_manager_rref: RRef of LocalActorManager on the node
        - gpus: currently available gpus on the node

    Methods:
        - `dispatch(gpu)`: dispatch `gpu` gpus on the node. dispatched gpus will be removed from `self.gpus`.
        - `recycle(resource)`: recycle `resource` on the node, adding gpus of `resource` back to `self.gpus`.
    """

    def __init__(self, name):
        self.name = name
        self.local_actor_manager_rref = rpc.remote(self.name, LocalActorManager)
        self.gpus = call_remote_actor(self.local_actor_manager_rref, "visible_gpus", (), {}).to_here()

    def dispatch(self, gpu) -> "NodeResource":
        assert len(self.gpus) >= gpu
        self.gpus.sort()
        gpus = self.gpus[:gpu]
        self.gpus = self.gpus[gpu:]
        ret = NodeResource(self, gpus)
        return ret

    def recycle(self, resource: "NodeResource"):
        assert resource.node == self
        self.gpus.extend(resource.gpus)


class NodeResource:
    """
    Resource on a node. Should be created by `Node.dispatch`.
    Create remote actors on the resource using `create_actor`.

    Members:
        - node: node of the resource
        - gpus: gpus of the resource
        - actors: list of RRef of actors created on the resource

    Methods:
        - `create_actor(cls, args, kwargs, env_vars, gpus)`: create an actor of `cls` on the resource.
    """

    def __init__(self, node: Node, gpus: List[int]):
        self.node = node
        self.gpus = gpus
        self.actors = []

    def create_actor(self, cls, args, kwargs, env_vars, gpus) -> rpc.RRef:
        assert all(i in self.gpus for i in gpus)
        rref = call_remote_actor(self.node.local_actor_manager_rref, "create_local_actor", (cls, args, kwargs, env_vars, gpus), {})
        self.actors.append(rref)
        return rref

    def __del__(self):
        for actor in self.actors:
            call_remote_actor(actor, "__del__", (), {})
        self.node.recycle(self)


class NodeManager:
    """
    Should be created on MASTER node to manage all remote nodes.

    Methods:
        - `init()`: initialize the node manager. Should be called on MASTER node.
        - `dispatch(gpu)`: dispatch `gpu` gpus on a node.
    """

    def __init__(self):
        self._initiated = False

    def init(self):
        if not self._initiated:
            self._initiated = True
            world_size = int(os.environ["TORCHRPC_WORLD_SIZE"])
            self.nodes = [Node(f"torchrpc_worker{i}") for i in range(world_size)]

    def dispatch(self, gpu) -> NodeResource:
        # TODO: CPU only下不应该全部派给一个node
        ret = None
        for node in self.nodes:
            if len(node.gpus) >= gpu:
                if ret is None or len(node.gpus) < len(ret.gpus):
                    ret = node
        assert ret is not None
        return ret.dispatch(gpu)


node_manager = NodeManager()
