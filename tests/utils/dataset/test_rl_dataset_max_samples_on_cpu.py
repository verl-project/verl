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
"""Tests for ``data.train_max_samples`` / ``data.val_max_samples`` subsampling.

The subset that ``max_samples`` draws must be identical across process starts,
otherwise a checkpoint resume (which only restores the sampler index) continues
on different rows than the run it resumes. See issue #7816.
"""

import pandas as pd
import pytest
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.ppo.utils import create_rl_sampler
from verl.utils.dataset.rl_dataset import DEFAULT_MAX_SAMPLES_SEED, RLHFDataset

TOTAL_ROWS = 64
MAX_SAMPLES = 8

# ``row_id``s that ``np.random.default_rng(seed).choice(64, size=8, replace=False)``
# draws. Pinned as literals so that silently changing ``DEFAULT_MAX_SAMPLES_SEED``
# away from 42 -- or dropping the fallback and drawing from entropy again -- fails
# here instead of passing against a recomputed expectation.
# (numpy only freezes ``RandomState`` streams, not ``Generator`` ones, so these
# literals also pin the numpy version the subset is reproducible under.)
SUBSET_DEFAULT_SEED = [62, 63, 53, 38, 60, 26, 5, 44]
EXPLICIT_SEED = 1234
SUBSET_EXPLICIT_SEED = [6, 16, 10, 57, 55, 22, 58, 56]


@pytest.fixture
def parquet_file(tmp_path):
    """Write a tiny parquet file whose rows are trivially identifiable."""
    path = tmp_path / "data.parquet"
    pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": f"question {i}"}] for i in range(TOTAL_ROWS)],
            "row_id": list(range(TOTAL_ROWS)),
        }
    ).to_parquet(path)
    return str(path)


def _data_config(tmp_path, **overrides):
    config = {
        "prompt_key": "prompt",
        # Avoid needing a real tokenizer: no tokenization happens in __init__.
        "filter_overlong_prompts": False,
        "cache_dir": str(tmp_path / "cache"),
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _row_ids(dataset):
    return list(dataset.dataframe["row_id"])


class _RowIdView:
    """Read-only view exposing each row's ``row_id`` in dataset order.

    ``RLHFDataset.__getitem__`` tokenizes, which needs a real tokenizer. The resume
    test only cares *which rows* the dataloader hands out, and the view preserves
    both the length and the order of the dataset it wraps.
    """

    def __init__(self, dataset):
        self._row_ids = _row_ids(dataset)

    def __len__(self):
        return len(self._row_ids)

    def __getitem__(self, index):
        return self._row_ids[index]


def _build_loader(parquet_file, config):
    """Build dataset + sampler + dataloader exactly as a fresh process would."""
    dataset = RLHFDataset(data_files=parquet_file, tokenizer=None, config=config, max_samples=MAX_SAMPLES)
    sampler = create_rl_sampler(config, dataset)
    return StatefulDataLoader(dataset=_RowIdView(dataset), batch_size=2, num_workers=0, sampler=sampler)


def _drain(batches):
    return [row_id for batch in batches for row_id in batch.tolist()]


def test_max_samples_subsample_is_reproducible_without_seed(parquet_file, tmp_path):
    """Without an explicit ``data.seed`` the drawn subset must still be stable."""
    config = _data_config(tmp_path, shuffle=True)

    first = RLHFDataset(data_files=parquet_file, tokenizer=None, config=config, max_samples=MAX_SAMPLES)
    second = RLHFDataset(data_files=parquet_file, tokenizer=None, config=config, max_samples=MAX_SAMPLES)

    assert len(first) == MAX_SAMPLES
    assert _row_ids(first) == _row_ids(second)
    assert _row_ids(first) == SUBSET_DEFAULT_SEED
    # A shuffled draw should not degenerate into the deterministic head slice.
    assert _row_ids(first) != list(range(MAX_SAMPLES))


def test_default_max_samples_seed_is_pinned():
    """The literals above are only meaningful while the fallback stays 42."""
    assert DEFAULT_MAX_SAMPLES_SEED == 42


def test_max_samples_subsample_honours_explicit_seed(parquet_file, tmp_path):
    """An explicitly configured seed keeps producing the subset it produces today."""
    config = _data_config(tmp_path, shuffle=True, seed=EXPLICIT_SEED)

    dataset = RLHFDataset(data_files=parquet_file, tokenizer=None, config=config, max_samples=MAX_SAMPLES)

    assert _row_ids(dataset) == SUBSET_EXPLICIT_SEED
    # An explicit seed must not collapse onto the fallback's subset.
    assert _row_ids(dataset) != SUBSET_DEFAULT_SEED


def test_max_samples_subsample_without_shuffle_is_head_slice(parquet_file, tmp_path):
    config = _data_config(tmp_path, shuffle=False)

    dataset = RLHFDataset(data_files=parquet_file, tokenizer=None, config=config, max_samples=MAX_SAMPLES)

    assert _row_ids(dataset) == list(range(MAX_SAMPLES))


def test_resume_continues_on_the_same_rows(parquet_file, tmp_path):
    """The actual bug: a resumed process must land on the rows it left off at.

    ``data.seed`` is deliberately unset -- the default, and the configuration that
    made ``max_samples`` redraw its subset on every process start. Only the sampler
    index is checkpointed, so a redrawn subset silently moves the run onto different
    rows even though the dataloader state restores perfectly.
    """
    config = _data_config(tmp_path, shuffle=True)

    loader = _build_loader(parquet_file, config)

    batches = iter(loader)
    consumed = _drain(next(batches) for _ in range(2))
    state = loader.state_dict()
    # What finishing this run without interruption would have produced.
    expected_remaining = _drain(batches)

    # A resumed process rebuilds dataset, sampler and dataloader from scratch.
    resumed_loader = _build_loader(parquet_file, config)
    resumed_loader.load_state_dict(state)
    actual_remaining = _drain(resumed_loader)

    assert len(consumed) == 4
    assert actual_remaining == expected_remaining
    # No row is replayed or skipped across the restart.
    assert not set(actual_remaining) & set(consumed)
    assert sorted(consumed + actual_remaining) == sorted(SUBSET_DEFAULT_SEED)
