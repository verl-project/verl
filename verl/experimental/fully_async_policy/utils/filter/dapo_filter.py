# Copyright 2025 Meituan Ltd. and/or its affiliates
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
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from verl import DataProto


def filter_dapo(new_batch: DataProto, config: DictConfig, batch: Optional[DataProto] = None, **kwargs) -> DataProto:
    """Filter DAPO batch by metric standard deviation.

    Since all uid values are identical, we directly compute the standard deviation
    of metrics across all samples without grouping logic.

    Args:
        new_batch: DataProto batch to filter
        config: Configuration object with algorithm.filter_groups.metric
        batch: Optional existing batch to concatenate with
        **kwargs: Additional arguments

    Returns:
        Filtered batch (either new_batch alone or concatenated with existing batch)
    """
    metric_name = config.algorithm.filter_groups.metric

    # Compute sequence-level metric
    if metric_name == "seq_final_reward":
        metric_vals = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
    elif metric_name == "seq_reward":
        metric_vals = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
    else:
        # Fallback: assume metric already exists in non_tensor_batch
        metric_vals = new_batch.non_tensor_batch.get(metric_name)
        if metric_vals is None:
            raise ValueError(f"Metric '{metric_name}' not found in batch")
        metric_vals = np.array(metric_vals) if not isinstance(metric_vals, np.ndarray) else metric_vals

    # Compute standard deviation across all samples
    metric_std = np.std(metric_vals)

    # Keep all samples if std > 0, or if there's only 1 sample
    if metric_std > 0 or len(metric_vals) == 1:
        # Keep all samples
        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
    else:
        # All metrics are identical and sample count > 1, filter out all samples
        new_batch = new_batch[[]]  # Empty batch
        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

    return batch
