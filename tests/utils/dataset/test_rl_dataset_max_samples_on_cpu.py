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

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import create_rl_dataset
from verl.utils.dataset.rl_dataset import RLHFDataset

TOTAL_ROWS = 64
MAX_SAMPLES = 8


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


def test_max_samples_subsample_is_reproducible_without_seed(parquet_file, tmp_path):
    """Without an explicit ``data.seed`` the drawn subset must still be stable."""
    config = _data_config(tmp_path, shuffle=True)

    first = RLHFDataset(data_files=parquet_file, tokenizer=None, config=config, max_samples=MAX_SAMPLES)
    second = RLHFDataset(data_files=parquet_file, tokenizer=None, config=config, max_samples=MAX_SAMPLES)

    assert len(first) == MAX_SAMPLES
    assert _row_ids(first) == _row_ids(second)
    # A shuffled draw should not degenerate into the deterministic head slice.
    assert _row_ids(first) != list(range(MAX_SAMPLES))


def test_max_samples_subsample_honours_explicit_seed(parquet_file, tmp_path):
    """An explicitly configured seed keeps producing the subset it produces today."""
    seed = 1234
    config = _data_config(tmp_path, shuffle=True, seed=seed)

    dataset = RLHFDataset(data_files=parquet_file, tokenizer=None, config=config, max_samples=MAX_SAMPLES)

    expected = np.random.default_rng(seed).choice(TOTAL_ROWS, size=MAX_SAMPLES, replace=False)
    assert _row_ids(dataset) == expected.tolist()


def test_max_samples_subsample_without_shuffle_is_head_slice(parquet_file, tmp_path):
    config = _data_config(tmp_path, shuffle=False)

    dataset = RLHFDataset(data_files=parquet_file, tokenizer=None, config=config, max_samples=MAX_SAMPLES)

    assert _row_ids(dataset) == list(range(MAX_SAMPLES))


def test_val_subsample_follows_validation_shuffle(parquet_file, tmp_path):
    """The validation subset must key off ``validation_shuffle``, not ``shuffle``."""
    config = _data_config(tmp_path, shuffle=True, validation_shuffle=False)

    train_dataset = create_rl_dataset(
        parquet_file, config, tokenizer=None, processor=None, is_train=True, max_samples=MAX_SAMPLES
    )
    val_dataset = create_rl_dataset(
        parquet_file, config, tokenizer=None, processor=None, is_train=False, max_samples=MAX_SAMPLES
    )

    assert _row_ids(val_dataset) == list(range(MAX_SAMPLES))
    assert _row_ids(train_dataset) != list(range(MAX_SAMPLES))
    # Building the val dataset must not mutate the shared data config.
    assert config.shuffle is True


def test_val_subsample_can_shuffle_while_train_does_not(parquet_file, tmp_path):
    config = _data_config(tmp_path, shuffle=False, validation_shuffle=True)

    train_dataset = create_rl_dataset(
        parquet_file, config, tokenizer=None, processor=None, is_train=True, max_samples=MAX_SAMPLES
    )
    val_dataset = create_rl_dataset(
        parquet_file, config, tokenizer=None, processor=None, is_train=False, max_samples=MAX_SAMPLES
    )

    assert _row_ids(train_dataset) == list(range(MAX_SAMPLES))
    assert _row_ids(val_dataset) != list(range(MAX_SAMPLES))
