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


from verl.single_controller.base.worker import Worker
from verl.single_controller.torchrpc.base import TorchRPCClassWithInitArgs, TorchRPCResourcePool, TorchRPCWorkerGroup, torchrpc_remote


class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    def __init__(self, cuda_visible_devices=None) -> None:
        super().__init__(cuda_visible_devices)


@torchrpc_remote
def test():
    # test single-node-no-partition
    print("test single-node-no-partition")
    resource_pool = TorchRPCResourcePool([8], use_gpu=True)

    class_with_args = TorchRPCClassWithInitArgs(cls=TestActor)

    print("create actor worker group")
    actor_wg = TorchRPCWorkerGroup(resource_pool, class_with_args)
    print("create critic worker group")
    critic_wg = TorchRPCWorkerGroup(resource_pool, class_with_args)
    print("create rm worker group")
    rm_wg = TorchRPCWorkerGroup(resource_pool, class_with_args)
    print("create ref worker group")
    ref_wg = TorchRPCWorkerGroup(resource_pool, class_with_args)

    assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    assert rm_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]

    del actor_wg
    del critic_wg
    del rm_wg
    del ref_wg

    del resource_pool

    # test single-node-multi-partition
    print("test single-node-multi-partition")
    rm_resource_pool = TorchRPCResourcePool([4], use_gpu=True)
    ref_resource_pool = TorchRPCResourcePool([4], use_gpu=True)

    assert rm_resource_pool.world_size == 4
    assert ref_resource_pool.world_size == 4

    # actor_wg = TorchRPCWorkerGroup(total_resource_pool, class_with_args)
    # critic_wg = TorchRPCWorkerGroup(total_resource_pool, class_with_args)
    rm_wg = TorchRPCWorkerGroup(rm_resource_pool, class_with_args)
    ref_wg = TorchRPCWorkerGroup(ref_resource_pool, class_with_args)

    # assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    # assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    assert rm_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(4)]
    assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(4, 8)]


if __name__ == "__main__":
    test()
