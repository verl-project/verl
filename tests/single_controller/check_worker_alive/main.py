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

import os
import sys
import time

import ray

from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


@ray.remote
class TestActor(Worker):
    def __init__(self) -> None:
        super().__init__()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def foo(self, wait_time):
        time.sleep(wait_time)
        sys.exit(1)


if __name__ == "__main__":
    wait_time = int(os.getenv("WAIT_TIME", "10"))

    ray.init()

    # test single-node-no-partition
    print("test single-node-no-partition")
    resource_pool = RayResourcePool([2], use_gpu=False)
    class_with_args = RayClassWithInitArgs(cls=TestActor)

    print("create worker group")
    wg = RayWorkerGroup(resource_pool, class_with_args, name_prefix="test")

    wg.start_worker_aliveness_check(1)
    time.sleep(1)

    print(time.time(), "start foo")

    _ = wg.foo(wait_time)
    print("foo started")

    print(
        time.time(),
        f"wait 6x wait time {wait_time * 6} to let signal returned to process but still not exceed process wait time",
    )
    time.sleep(wait_time * 6)

    ray.shutdown()
