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

"""Group-ratio sampler for balanced data mixing.

Groups dataset indices by a configurable key (e.g., ``data_source``,
``extra_info.label``) and yields batches with configurable per-group
ratios.  Compatible with ``torchdata.stateful_dataloader.StatefulDataLoader``
via ``state_dict`` / ``load_state_dict``.

Example config::

    data:
      sampler:
        class_path: verl.utils.dataset.group_ratio_sampler
        class_name: GroupRatioSampler
        group_key: data_source
        group_names: ["openai/gsm8k", "lighteval/MATH"]
        group_ratios: [3, 7]
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _resolve_dot_path(obj, dot_path: str):
    """Resolve a dot-separated path like ``extra_info.csnvList.0.label``.

    Supports dict keys, list indices, and object attributes.
    Returns ``None`` if any segment is missing.
    """
    parts = dot_path.split(".")
    cur = obj
    for part in parts:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list | tuple):
            try:
                idx = int(part)
            except ValueError:
                return None
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        else:
            cur = getattr(cur, part, None)
    return cur


class GroupRatioSampler:
    """Sampler that yields indices with configurable per-group ratios.

    Groups dataset indices by extracting a group value from each row using
    ``group_key`` (a dot-separated path into the parquet row).  Each batch
    contains ``per_group_counts[name]`` indices from each group, with
    independent wrap-around so minority groups are oversampled without
    forcing majority groups to discard data.

    The sampler is stateful: ``state_dict`` / ``load_state_dict`` save and
    restore the shuffled index arrays, cursors, RNG state, and epoch count
    for reproducible checkpoint resumption.
    """

    def __init__(
        self,
        data_source,
        data_config=None,
        *,
        group_key: str = "data_source",
        group_names=None,
        group_ratios=None,
        seed=None,
        **kwargs,
    ):
        self.data_source = data_source
        self.group_key = group_key
        self.seed = seed

        # Resolve config overrides when called via create_rl_sampler
        if data_config is not None:
            sampler_cfg = data_config.get("sampler", {})
            group_key = sampler_cfg.get("group_key", group_key)
            group_names = sampler_cfg.get("group_names", group_names)
            group_ratios = sampler_cfg.get("group_ratios", group_ratios)
            seed = sampler_cfg.get("seed", seed)
        # Persist the resolved values on self so the config overrides take effect
        self.group_key = group_key
        self.seed = seed

        if group_names is None or group_ratios is None:
            raise ValueError(
                "GroupRatioSampler requires group_names and group_ratios. "
                "Set them in config.data.sampler.group_names / group_ratios."
            )

        self.group_names = list(group_names)
        self.group_ratios = list(group_ratios)
        self.num_groups = len(self.group_names)

        assert len(self.group_ratios) == self.num_groups, (
            f"group_ratios length {len(self.group_ratios)} != group_names length {self.num_groups}"
        )
        assert all(r > 0 for r in self.group_ratios), "group_ratios must all be positive"

        # Resolve batch_size from config or dataset
        if data_config is not None:
            self.batch_size = data_config.get("train_batch_size", 256)
        else:
            self.batch_size = kwargs.get("batch_size", 256)

        # Largest-remainder method for per-group counts
        total_ratio = sum(self.group_ratios)
        raw = [self.batch_size * r / total_ratio for r in self.group_ratios]
        floor = [int(c) for c in raw]
        frac = [c - f for c, f in zip(raw, floor, strict=False)]
        allocated = sum(floor)
        remainder = self.batch_size - allocated
        for i in sorted(range(self.num_groups), key=lambda j: -frac[j])[:remainder]:
            floor[i] += 1
        self.per_group_counts = {name: floor[i] for i, name in enumerate(self.group_names)}

        for name, count in self.per_group_counts.items():
            logger.info(
                f"GroupRatioSampler: '{name}' "
                f"ratio={self.group_ratios[self.group_names.index(name)]}, "
                f"per_batch={count}"
            )

        self.rng = np.random.default_rng(seed)

        # Group indices by group value
        self.group_to_indices: dict[str, list[int]] = {name: [] for name in self.group_names}
        self._group_indices()

        for name in self.group_names:
            count = len(self.group_to_indices[name])
            logger.info(f"GroupRatioSampler: '{name}' has {count} samples in dataset")
            if count == 0:
                raise ValueError(
                    f"GroupRatioSampler: group '{name}' has 0 samples. "
                    f"Check that group_key='{self.group_key}' resolves to "
                    f"'{name}' for at least some rows in the dataset."
                )

        self._epoch_count = 0
        self._reset_epoch()

    def _group_indices(self):
        """Iterate the dataset once and group indices by group value."""
        dataframe = self.data_source.dataframe
        total = len(dataframe)
        logger.info(f"GroupRatioSampler: grouping {total} samples by '{self.group_key}'...")

        try:
            column = dataframe[self.group_key.split(".")[0]]
            use_column = True
        except Exception:
            use_column = False

        if use_column and "." not in self.group_key:
            # Simple column access (e.g., "data_source")
            for i, value in enumerate(column):
                value_str = str(value) if value is not None else None
                if value_str in self.group_to_indices:
                    self.group_to_indices[value_str].append(i)
                elif value_str is not None:
                    logger.warning(f"GroupRatioSampler: unknown group '{value_str}' at index {i}, skipping")
        else:
            # Dot-path access (e.g., "extra_info.csnvList.0.label")
            for i in range(total):
                row = dataframe[i]
                value = _resolve_dot_path(row, self.group_key)
                value_str = str(value) if value is not None else None
                if value_str in self.group_to_indices:
                    self.group_to_indices[value_str].append(i)
                elif value_str is not None:
                    logger.warning(f"GroupRatioSampler: unknown group '{value_str}' at index {i}, skipping")

    def _reset_epoch(self):
        """Shuffle each group's indices independently for a new epoch."""
        self._shuffled: dict[str, np.ndarray] = {}
        for name in self.group_names:
            arr = np.array(self.group_to_indices[name], dtype=np.int64)
            self.rng.shuffle(arr)
            self._shuffled[name] = arr
        self._cursors: dict[str, int] = {name: 0 for name in self.group_names}
        self._epoch_count += 1
        logger.info(f"GroupRatioSampler: starting epoch {self._epoch_count}")

    def __iter__(self):
        """Yield all indices for one epoch with per-group ratio distribution.

        Each ``next()`` call on the returned iterator produces one batch
        (``batch_size`` indices with the configured group ratio), repeated
        ``len(dataset) // batch_size`` times to cover one epoch.
        Minority groups wrap around independently; majority groups are
        exhausted exactly once per epoch.
        """
        num_batches = len(self.data_source) // self.batch_size
        for _ in range(num_batches):
            batch: list[int] = []

            for name in self.group_names:
                indices = self._shuffled[name]
                cursor = self._cursors[name]
                n = len(indices)
                if n == 0:
                    continue

                taken: list[int] = []
                remaining = self.per_group_counts[name]
                while remaining > 0:
                    take = min(remaining, n - cursor)
                    taken.extend(indices[cursor : cursor + take].tolist())
                    remaining -= take
                    cursor += take
                    if cursor >= n:
                        # Independent wrap-around: reshuffle only this group
                        arr = np.array(self.group_to_indices[name], dtype=np.int64)
                        self.rng.shuffle(arr)
                        self._shuffled[name] = arr
                        cursor = 0
                        logger.info(
                            f"GroupRatioSampler: group '{name}' exhausted, reshuffled (epoch {self._epoch_count})"
                        )

                batch.extend(taken)
                self._cursors[name] = cursor

            # Shuffle the final batch to mix groups
            batch_arr = np.array(batch, dtype=np.int64)
            self.rng.shuffle(batch_arr)
            yield from batch_arr.tolist()

    def __len__(self):
        return len(self.data_source)

    def state_dict(self) -> dict:
        """Return sampler state for checkpoint resumption."""
        return {
            "shuffled": {k: v.tolist() for k, v in self._shuffled.items()},
            "cursors": dict(self._cursors),
            "rng_state": self.rng.bit_generator.state,
            "epoch_count": self._epoch_count,
        }

    def load_state_dict(self, state: dict):
        """Restore sampler state from a checkpoint."""
        self._shuffled = {k: np.array(v, dtype=np.int64) for k, v in state["shuffled"].items()}
        self._cursors = dict(state["cursors"])
        self.rng.bit_generator.state = state["rng_state"]
        self._epoch_count = state["epoch_count"]
        logger.info(f"GroupRatioSampler: restored state (epoch {self._epoch_count}, cursors {self._cursors})")
