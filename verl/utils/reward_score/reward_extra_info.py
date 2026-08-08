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

"""Batch assembly helpers for per-sample reward metadata."""

import numpy as np


def assemble_reward_extra_info(reward_extra_infos: list[dict]) -> dict[str, np.ndarray]:
    """Align per-sample reward metadata into batch-level arrays.

    Keys are collected from every sample in stable first-seen order. A key that
    is absent from any sample is filled with ``None`` and stored in an object
    array; fully populated keys retain NumPy's natural inferred dtype.

    This helper only assembles and aligns columns. Downstream metric handling is
    responsible for interpreting the resulting values, including ``None``.
    """
    keys: list[str] = []
    seen: set[str] = set()
    for info in reward_extra_infos:
        for key in info:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    non_tensor_batch = {}
    for key in keys:
        values = [info.get(key) for info in reward_extra_infos]
        if all(key in info for info in reward_extra_infos):
            non_tensor_batch[key] = np.array(values)
        else:
            non_tensor_batch[key] = np.array(values, dtype=object)

    return non_tensor_batch
