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

import pytest
import ray
import torch


@ray.remote(num_cpus=1, num_gpus=1)
class WeightWorker:
    def __init__(self, trainer, master, shared_visibility, visible_devices, rebuild_group):
        from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

        if shared_visibility:
            assigned = str(ray.get_gpu_ids()[0])
            if assigned.replace(".", "", 1).isdigit():
                assigned = str(int(float(assigned)))
            self.device = visible_devices.index(assigned)
        else:
            self.device = 0
        torch.cuda.set_device(self.device)
        self.trainer = trainer
        self.engine = NCCLCheckpointEngine(
            bucket_size=64,
            is_master=master,
            rebuild_group=rebuild_group,
            multi_sender=True,
        )

    def prepare(self):
        return self.engine.prepare()

    def initialize(self, **kwargs):
        self.engine.init_process_group(**kwargs)

    async def transfer(self, step):
        if self.trainer:
            weight = torch.arange(33, dtype=torch.float32, device=f"cuda:{self.device}") + step
            await self.engine.send_weights(iter([("weight", weight)]), global_steps=step)
            return None
        result = {}
        async for name, weight in self.engine.receive_weights(global_steps=step):
            result[name] = weight.cpu().clone()
        return result

    def finalize(self):
        self.engine.finalize()

    def close(self):
        self.engine.rebuild_group = True
        self.engine.finalize()


@pytest.mark.parametrize("shared_visibility", [False, True])
@pytest.mark.parametrize("rebuild_group", [False, True])
@pytest.mark.parametrize("trainers", [1, 2])
def test_weight_sync_across_device_visibility_layouts(shared_visibility, rebuild_group, trainers):
    if torch.cuda.device_count() < trainers + 1:
        pytest.skip(f"requires {trainers + 1} CUDA devices")
    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible == [""]:
        visible = [str(index) for index in range(torch.cuda.device_count())]
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(include_dashboard=False)
    runtime_env = {"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}} if shared_visibility else {}
    workers = [
        WeightWorker.options(runtime_env=runtime_env).remote(
            rank < trainers,
            rank == 0,
            shared_visibility,
            visible,
            rebuild_group,
        )
        for rank in range(trainers + 1)
    ]
    try:
        for step in range(3):
            metadata = ray.get([worker.prepare.remote() for worker in workers])
            sender, receiver = NCCLCheckpointEngine.build_topology(trainers, 1, metadata)
            options = [{key: values[rank] for key, values in sender.items()} for rank in range(trainers)]
            options.append({key: values[0] for key, values in receiver.items()})
            ray.get(
                [worker.initialize.remote(**values) for worker, values in zip(workers, options, strict=True)],
                timeout=120,
            )
            result = ray.get([worker.transfer.remote(step) for worker in workers], timeout=120)
            torch.testing.assert_close(result[-1]["weight"], torch.arange(33, dtype=torch.float32) + step)
            ray.get([worker.finalize.remote() for worker in workers], timeout=120)
        ray.get([worker.close.remote() for worker in workers], timeout=120)
    finally:
        for worker in workers:
            ray.kill(worker)
        if owns_ray:
            ray.shutdown()
