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

import os

import pytest
import torch
from torch.utils.data import DistributedSampler, TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.utils.checkpoint import CheckpointHandler, OrchestrationMode


class _FakeEngine:
    def save_checkpoint(self, local_path, global_step, max_ckpt_to_keep=None):
        os.makedirs(local_path, exist_ok=True)

    def load_checkpoint(self, local_path):
        assert os.path.isdir(local_path)

    def is_mp_src_rank_with_outputs(self):
        return True

    def get_data_parallel_rank(self):
        return 0


def _make_loader(dataset_size=8, batch_size=2):
    dataset = TensorDataset(torch.arange(dataset_size))
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=True)
    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    return sampler, loader


def _batch_values(loader):
    return [batch[0].tolist() for batch in loader]


def _save_after_batches(checkpoint_dir, batches, *, dataset_size=8, batch_size=2, mode=OrchestrationMode.RAY):
    _, loader = _make_loader(dataset_size=dataset_size, batch_size=batch_size)
    iterator = iter(loader)
    consumed = [next(iterator)[0].tolist() for _ in range(batches)]
    handler = CheckpointHandler(
        engine=_FakeEngine(),
        train_dataloader=loader,
        default_local_dir=checkpoint_dir,
        resume_mode="disable",
        mode=mode,
    )
    handler.save_checkpoint(step=batches)
    remaining = [batch[0].tolist() for batch in iterator]
    return consumed, remaining


def _resume(checkpoint_dir, *, dataset_size=8, batch_size=2, mode=OrchestrationMode.RAY):
    sampler, loader = _make_loader(dataset_size=dataset_size, batch_size=batch_size)
    handler = CheckpointHandler(
        engine=_FakeEngine(),
        train_dataloader=loader,
        default_local_dir=checkpoint_dir,
        resume_mode="auto",
        mode=mode,
    )
    resume_step = handler.load_checkpoint()
    sampler.set_epoch(resume_step // len(loader))
    return resume_step, _batch_values(loader)


@pytest.mark.parametrize("mode", [OrchestrationMode.RAY, OrchestrationMode.SPMD])
def test_epoch_boundary_resume_starts_next_sft_epoch(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    checkpoint_dir = str(tmp_path / "boundary")
    _, remaining = _save_after_batches(checkpoint_dir, batches=4, mode=mode)
    assert remaining == []

    fresh_sampler, fresh_loader = _make_loader()
    fresh_sampler.set_epoch(1)
    expected_next_epoch = _batch_values(fresh_loader)

    resume_step, resumed_batches = _resume(checkpoint_dir, mode=mode)

    assert resume_step == 4
    assert resumed_batches == expected_next_epoch


@pytest.mark.parametrize("mode", [OrchestrationMode.RAY, OrchestrationMode.SPMD])
def test_mid_epoch_resume_restores_sft_dataloader_position(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    checkpoint_dir = str(tmp_path / "mid_epoch")
    consumed, expected_remaining = _save_after_batches(checkpoint_dir, batches=2, mode=mode)

    resume_step, resumed_batches = _resume(checkpoint_dir, mode=mode)

    assert resume_step == 2
    assert len(consumed) == 2
    assert resumed_batches == expected_remaining
