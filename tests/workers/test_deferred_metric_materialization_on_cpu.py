# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl.utils.metric import Metric
from verl.workers.engine_workers import TrainingWorker


class _DeferredScalar(torch.Tensor):
    item_calls = 0
    cpu_calls = 0

    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, torch.tensor(value), require_grad=False)

    @property
    def device(self):
        return torch.device("cuda:0")

    def item(self):
        type(self).item_calls += 1
        return self.as_subclass(torch.Tensor).item()

    def cpu(self):
        type(self).cpu_calls += 1
        return self.as_subclass(torch.Tensor)

    @classmethod
    def reset_calls(cls):
        cls.item_calls = 0
        cls.cpu_calls = 0


def _contains_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Metric):
        return any(_contains_tensor(metric_value) for metric_value in value.values)
    if isinstance(value, dict):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_tensor(child) for child in value)
    return False


def test_postprocess_batches_loss_and_metric_materialization_before_object_gather():
    _DeferredScalar.reset_calls()
    worker = object.__new__(TrainingWorker)
    worker.device_name = "cpu"
    worker.engine = SimpleNamespace(get_data_parallel_group=lambda: object())
    worker.flops_counter = None
    metric = Metric("mean")
    metric.extend([_DeferredScalar(3.0), _DeferredScalar(5.0)])
    output = {
        "loss": [_DeferredScalar(1.0), _DeferredScalar(2.0)],
        "metrics": {
            "metric": metric,
            "plain": [_DeferredScalar(7.0)],
        },
        "model_output": {},
    }
    captured_metrics = None

    def fake_allgather(data, group):
        nonlocal captured_metrics
        captured_metrics = data
        assert not _contains_tensor(data)
        return {key: [value] for key, value in data.items()}

    device = SimpleNamespace(
        max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0,
    )
    with (
        patch("verl.workers.engine_workers.torch.distributed.all_reduce"),
        patch("verl.workers.engine_workers.allgather_dict_into_dict", side_effect=fake_allgather),
        patch("verl.workers.engine_workers.get_torch_device", return_value=device),
    ):
        result = worker._postprocess_output(
            output,
            global_token_num=None,
            delta_time=1.0,
            forward_only=False,
            images_seqlens=None,
        )

    assert captured_metrics is not None
    assert captured_metrics["metric"].values == [3.0, 5.0]
    assert captured_metrics["plain"] == [7.0]
    assert _DeferredScalar.item_calls == 0
    assert _DeferredScalar.cpu_calls == 1
    result_metrics = result.get_non_tensor("metrics")
    assert result_metrics["loss"] == 3.0
