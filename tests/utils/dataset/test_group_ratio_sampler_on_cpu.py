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
"""Unit tests for GroupRatioSampler.

Covers per-group ratio composition, oversampling of imbalanced pools,
Largest-remainder allocation, checkpoint resumption via ``state_dict`` and
the config-driven wiring in ``create_rl_sampler``. All tests run on CPU.
"""

import pandas as pd

from verl.trainer.ppo.utils import create_rl_sampler
from verl.utils.dataset.group_ratio_sampler import GroupRatioSampler


class _FakeData:
    """Minimal stand-in for a verl dataset: exposes ``dataframe`` and ``__len__``."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)


class _Cfg:
    """DictConfig-like object supporting both ``.get`` and attribute access."""

    def __init__(self, **values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)

    def __getattr__(self, key):
        values = object.__getattribute__(self, "_values")
        if key not in values:
            raise AttributeError(key)
        return values[key]


def _make_data(values):
    return _FakeData(pd.DataFrame({"data_source": values}))


def _sampler(data, group_names, group_ratios, batch_size, seed=None):
    return GroupRatioSampler(
        data_source=data,
        data_config=_Cfg(train_batch_size=batch_size),
        group_key="data_source",
        group_names=group_names,
        group_ratios=group_ratios,
        seed=seed,
    )


def _group_of(index, data):
    return data.dataframe.iloc[index]["data_source"]


def test_largest_remainder_allocation():
    """Per-batch counts follow Largest-remainder for ratios that do not divide evenly."""
    data = _make_data(["a"] * 10 + ["b"] * 100)
    sampler = _sampler(data, ["a", "b"], [5, 3], batch_size=16)
    assert sampler.per_group_counts == {"a": 10, "b": 6}


def test_per_batch_ratio_composition():
    """Every batch contains exactly ratio*size/sum indices from each group."""
    data = _make_data(["a"] * 30 + ["b"] * 70)
    sampler = _sampler(data, ["a", "b"], [3, 7], batch_size=10)
    assert sampler.per_group_counts == {"a": 3, "b": 7}

    it = iter(sampler)
    n_batches = len(data) // 10
    seen = []
    for _ in range(n_batches):
        batch = [next(it) for _ in range(10)]
        assert sum(1 for i in batch if _group_of(i, data) == "a") == 3
        assert sum(1 for i in batch if _group_of(i, data) == "b") == 7
        seen.extend(batch)

    # Balanced pools are exhausted exactly once per epoch: no duplicates.
    assert sorted(seen) == list(range(len(data)))


def test_minority_pool_wrap_around():
    """Minority group is oversampled via wrap-around between batches."""
    # 40 rows, 4 batches of 10; group "a" only has 3 rows but needs 3 per batch,
    # so its pool is exhausted (and reshuffled) at every batch boundary.
    data = _make_data(["a"] * 3 + ["b"] * 37)
    sampler = _sampler(data, ["a", "b"], [3, 7], batch_size=10, seed=1)

    it = iter(sampler)
    all_a = []
    all_b = []
    for _ in range(len(data) // 10):
        batch = [next(it) for _ in range(10)]
        a_indices = [i for i in batch if _group_of(i, data) == "a"]
        b_indices = [i for i in batch if _group_of(i, data) == "b"]
        assert len(a_indices) == 3
        assert len(b_indices) == 7
        all_a.extend(a_indices)
        all_b.extend(b_indices)

    # Minority pool (3 rows) is fully reused in every batch: 4 uses per row.
    assert sorted(all_a) == sorted([0, 1, 2] * 4)
    # Majority pool (37 rows) needs 28 rows over the epoch: no wrap, all distinct.
    assert len(set(all_b)) == 28
    assert all(3 <= i < 40 for i in all_b)


def test_state_dict_resumption():
    """Restoring state mid-epoch reproduces the exact remaining index stream."""
    data = _make_data(["a"] * 30 + ["b"] * 70)
    sampler = _sampler(data, ["a", "b"], [3, 7], batch_size=10, seed=0)
    it = iter(sampler)
    consumed = [next(it) for _ in range(30)]
    assert len(consumed) == 30

    state = sampler.state_dict()
    assert set(state) == {"shuffled", "cursors", "rng_state", "epoch_count"}

    restored = _sampler(data, ["a", "b"], [3, 7], batch_size=10, seed=0)
    restored.load_state_dict(state)
    it_restored = iter(restored)

    rest = [next(it) for _ in range(10)]
    rest_restored = [next(it_restored) for _ in range(10)]
    assert rest == rest_restored


def test_create_rl_sampler_wiring():
    """config.data.sampler.class_path drives the custom sampler via create_rl_sampler."""
    data = _make_data(["a"] * 30 + ["b"] * 70)
    sampler_cfg = _Cfg(
        class_path="pkg://verl.utils.dataset.group_ratio_sampler",
        class_name="GroupRatioSampler",
        group_key="data_source",
        group_names=["a", "b"],
        group_ratios=[3, 7],
        seed=42,
    )
    data_cfg = _Cfg(sampler=sampler_cfg, train_batch_size=10, shuffle=True)

    sampler = create_rl_sampler(data_cfg, data)
    assert isinstance(sampler, GroupRatioSampler)
    assert sampler.per_group_counts == {"a": 3, "b": 7}


def _make_nested_data(rows):
    """Build a DataFrame whose ``extra_info`` column holds dict-like cells."""
    return _FakeData(pd.DataFrame({"extra_info": rows}))


def test_dot_path_group_key_resolves_into_nested_cells():
    """group_key='extra_info.label' reads the top-level column then resolves
    the remaining dot-path against each cell value (dict access), not against
    the DataFrame itself (which would return a column, not a row)."""
    rows = [{"label": "a"}] * 30 + [{"label": "b"}] * 70
    data = _make_nested_data(rows)
    sampler = GroupRatioSampler(
        data_source=data,
        data_config=_Cfg(train_batch_size=10),
        group_key="extra_info.label",
        group_names=["a", "b"],
        group_ratios=[3, 7],
        seed=0,
    )
    assert sampler.per_group_counts == {"a": 3, "b": 7}
    assert len(sampler.group_to_indices["a"]) == 30
    assert len(sampler.group_to_indices["b"]) == 70

    it = iter(sampler)
    for _ in range(len(data) // 10):
        batch = [next(it) for _ in range(10)]
        cells = [data.dataframe["extra_info"].iloc[i] for i in batch]
        assert sum(1 for c in cells if c["label"] == "a") == 3
        assert sum(1 for c in cells if c["label"] == "b") == 7


def test_dot_path_with_list_index_segment():
    """A numeric segment in the dot-path indexes into a list cell, e.g.
    'extra_info.items.0.kind' resolves items[0].kind on each cell."""
    rows = [
        {"items": [{"kind": "a"}]},
        {"items": [{"kind": "b"}]},
    ] * 50  # 50 'a' rows and 50 'b' rows, interleaved
    data = _make_nested_data(rows)
    sampler = GroupRatioSampler(
        data_source=data,
        data_config=_Cfg(train_batch_size=10),
        group_key="extra_info.items.0.kind",
        group_names=["a", "b"],
        group_ratios=[1, 1],
        seed=0,
    )
    assert len(sampler.group_to_indices["a"]) == 50
    assert len(sampler.group_to_indices["b"]) == 50


def test_dot_path_head_must_be_a_column():
    """A group_key whose head is not a column raises ValueError at init."""
    data = _make_nested_data([{"label": "a"}] * 10)
    try:
        GroupRatioSampler(
            data_source=data,
            data_config=_Cfg(train_batch_size=10),
            group_key="nonexistent.label",
            group_names=["a"],
            group_ratios=[1],
            seed=0,
        )
    except ValueError as e:
        assert "nonexistent" in str(e)
    else:
        raise AssertionError("expected ValueError for missing group_key head")


def test_len_matches_iter_yield_count():
    """__len__ equals the number of indices the iterator actually yields
    (drop_last), not len(dataset), when the dataset is not divisible by
    batch_size."""
    data = _make_data(["a"] * 33 + ["b"] * 67)  # 100 rows, batch_size=16 -> 96
    sampler = _sampler(data, ["a", "b"], [3, 7], batch_size=16)
    assert len(sampler) == 96
    assert len(list(iter(sampler))) == 96
