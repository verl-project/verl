# Copyright 2026 Individual Contributor: Jinhong Lin
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
"""Resuming SFT from a checkpoint must not perturb the restored global CPU RNG.

The checkpoint manager saves and restores ``torch.get_rng_state()``. Building the train
dataloader iterator after that restore must not draw from the global CPU RNG, otherwise
the resumed run continues from a different RNG state than an uninterrupted run.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf

from verl.trainer.sft_trainer import SFTTrainer


def _build_trainer(dp_rank=0, dp_size=1):
    config = OmegaConf.create(
        {
            "data": {"train_batch_size": 4 * dp_size, "pad_mode": "no_padding", "num_workers": 0},
            "trainer": {"seed": 1},
        }
    )
    engine = MagicMock()
    engine.get_data_parallel_rank.return_value = dp_rank
    engine.get_data_parallel_size.return_value = dp_size
    trainer = SimpleNamespace(config=config, engine=engine, train_dataset=list(range(64)), val_dataset=list(range(32)))
    SFTTrainer._build_dataloader(trainer)
    # The collator is not under test; batches of ints keep the comparison simple.
    trainer.train_dataloader.collate_fn = torch.utils.data.default_collate
    trainer.val_dataloader.collate_fn = torch.utils.data.default_collate
    return trainer


def test_resume_preserves_batches_and_global_cpu_rng():
    torch.manual_seed(0)
    trainer = _build_trainer()
    it = iter(trainer.train_dataloader)
    for _ in range(3):
        next(it)
    dataloader_state, rng_state = trainer.train_dataloader.state_dict(), torch.get_rng_state()
    expected_batches = [next(it).tolist() for _ in range(2)]
    expected_draws = torch.rand(4)

    # A fresh process: rebuild the dataloader, restore both states, then build the iterator.
    torch.manual_seed(12345)
    resumed = _build_trainer()
    resumed.train_dataloader.load_state_dict(dataloader_state)
    torch.set_rng_state(rng_state)
    resumed_it = iter(resumed.train_dataloader)

    assert [next(resumed_it).tolist() for _ in range(2)] == expected_batches
    assert torch.equal(torch.rand(4), expected_draws)


def test_building_iterators_does_not_consume_global_cpu_rng():
    trainer = _build_trainer()
    torch.manual_seed(0)
    before = torch.get_rng_state()
    iter(trainer.train_dataloader)
    iter(trainer.val_dataloader)
    assert torch.equal(torch.get_rng_state(), before)


def test_generators_are_seeded_per_dp_rank_and_split_train_from_val():
    seeds = []
    for dp_rank in range(2):
        trainer = _build_trainer(dp_rank=dp_rank, dp_size=2)
        seeds.append(trainer.train_dataloader.generator.initial_seed())
        seeds.append(trainer.val_dataloader.generator.initial_seed())
    assert len(set(seeds)) == len(seeds)
