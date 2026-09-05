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

from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.sft_val_utils import reduce_sft_val_loss, resolve_sft_val_batch_size


class _Toy(Dataset):
    def __len__(self):
        return 200

    def __getitem__(self, i):
        return i


def test_resolve_prefers_explicit_val_batch_size():
    assert resolve_sft_val_batch_size({"val_batch_size": 16}, 200) == 16


def test_resolve_defaults_to_full_val_set():
    assert resolve_sft_val_batch_size({}, 200) == 200
    assert resolve_sft_val_batch_size({"micro_batch_size_per_gpu": 4}, 200) == 200


def test_reduce_sft_val_loss_is_sample_weighted():
    # Batches of 4, 4, 4, 1 must not treat the tail as 25% of the mean.
    assert reduce_sft_val_loss([(1.0, 4), (1.0, 4), (1.0, 4), (13.0, 1)]) == (3 * 4 * 1.0 + 13.0) / 13
    assert reduce_sft_val_loss([]) is None


def _make_val_loader(*, drop_last: bool) -> DataLoader:
    dataset = _Toy()
    return StatefulDataLoader(
        dataset,
        batch_size=256,
        sampler=DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False, drop_last=drop_last),
        drop_last=drop_last,
    )


def test_drop_last_true_with_train_batch_is_empty():
    assert len(_make_val_loader(drop_last=True)) == 0


def test_drop_last_false_keeps_short_val_set():
    """Regression test for #7464: 200 samples, train-sized batch 256 must still yield a batch."""
    loader = _make_val_loader(drop_last=False)
    assert len(loader) >= 1
    seen = sum(len(batch) for batch in loader)
    assert seen == 200
