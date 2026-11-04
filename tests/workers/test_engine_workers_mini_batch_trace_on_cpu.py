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

"""How mini-batches show up in a trace of the engine workers.

The update loop names each iteration ``mini_batch<i>`` and advances the profiler once per
mini-batch: a mini-batch is the unit a ``torch.profiler.schedule`` sub-samples the update loop by
(see ``TorchProfilerScheduleConfig``). The forward-only stages (log-prob / ref / values) run their
batches without advancing the profiler, so a schedule never sub-samples them.

These tests drive the worker methods with mocked engines, so no GPU or ray is needed.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from tensordict import TensorDict

from verl.trainer.ppo.padding_utils import PADDING_MASK_KEY
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import TrainingWorker


def _engine(**overrides):
    engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(),
        eval_mode=lambda **kwargs: nullcontext(),
        infer_batch=lambda data, loss_function=None: {},
        is_mp_src_rank_with_outputs=lambda: False,
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_size=lambda: 1,
    )
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


def _worker(engine):
    worker = TrainingWorker.__new__(TrainingWorker)
    worker.engine = engine
    worker.engine_config = SimpleNamespace(
        forward_only=False,
        use_dynamic_bsz=False,
        infer_max_token_len_per_gpu=128,
        infer_micro_batch_size_per_gpu=1,
        max_token_len_per_gpu=128,
        micro_batch_size_per_gpu=1,
        use_fused_kernels=False,
    )
    worker.model_config = {}
    worker.loss_fn = lambda: None
    worker.profiler = MagicMock()
    return worker


def _record_names(monkeypatch):
    """Collect the names of the record_function ranges the worker opens."""
    names = []

    def fake_record_function(name):
        names.append(name)
        return nullcontext()

    monkeypatch.setattr(torch.profiler, "record_function", fake_record_function)
    return names


def test_worker_layout_updates_profiler_and_scheduler(monkeypatch):
    mini_batches = [TensorDict({}, batch_size=[]) for _ in range(3)]
    monkeypatch.setattr(tu, "make_iterator", lambda data, **kwargs: iter(mini_batches))
    names = _record_names(monkeypatch)

    worker = _worker(_engine())
    scheduler_updates = []
    worker.train_batch = lambda batch: scheduler_updates.append(tu.get(batch, "update_lr_scheduler")) or {}

    data = TensorDict({}, batch_size=[3])
    tu.assign_non_tensor(data, mini_batch_size=1)

    assert TrainingWorker.train_mini_batch(worker, data) is None
    # Numbered from the start of the step: without the index every iteration looks alike.
    assert names == ["mini_batch0", "mini_batch1", "mini_batch2"]
    # The profiler is advanced once per mini-batch -- the unit a torch.profiler.schedule
    # sub-samples the update loop by.
    assert worker.profiler.step.call_count == len(mini_batches)
    assert scheduler_updates == [False, False, True]


def test_controller_layout_excludes_padding_from_loss_normalization():
    worker = _worker(_engine(get_data_parallel_size=lambda: 2))
    seen_batches = []

    def train_batch(batch):
        seen_batches.append(
            (batch["row"].tolist(), tu.get(batch, "global_batch_size"), tu.get(batch, "update_lr_scheduler"))
        )
        return {}

    worker.train_batch = train_batch
    data = TensorDict({"row": torch.arange(5)}, batch_size=[5])
    tu.assign_non_tensor_data(data, "global_mini_batch_sizes", [4, 6])
    tu.assign_non_tensor_data(data, "global_mini_batch_padding_counts", [1, 1])

    assert TrainingWorker.train_mini_batch(worker, data) is None
    assert seen_batches == [
        ([0, 1], 3, False),
        ([2, 3, 4], 5, True),
    ]


def test_worker_layout_counts_shuffled_non_padding_rows_across_dp(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def all_gather_object(output, local_valid_rows, group):
        output[:] = [local_valid_rows, 2]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    worker = _worker(
        _engine(
            get_data_parallel_size=lambda: 2,
            get_data_parallel_group=lambda: object(),
        )
    )
    seen_batches = []
    worker.train_batch = (
        lambda batch: seen_batches.append((batch["row"].tolist(), tu.get(batch, "global_batch_size"))) or {}
    )
    data = TensorDict({"row": torch.arange(4)}, batch_size=[4])
    tu.assign_non_tensor(
        data,
        mini_batch_size=4,
        global_batch_size=4,
        dataloader_kwargs={"shuffle": True},
    )
    tu.assign_non_tensor_data(data, PADDING_MASK_KEY, [False, True, False, False] * 2)

    assert TrainingWorker.train_mini_batch(worker, data) is None
    assert len(seen_batches) == 2
    assert sum(global_batch_size for _, global_batch_size in seen_batches) == 7
    for rows, global_batch_size in seen_batches:
        assert global_batch_size == sum(row != 1 for row in rows) + 2


def test_forward_only_stage_is_not_a_profiler_step():
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, global_token_num=[4])
    worker = _worker(_engine())

    assert TrainingWorker.infer_batch(worker, data) is None
    worker.profiler.step.assert_not_called()
