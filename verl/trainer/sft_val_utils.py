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

from typing import Optional


def resolve_sft_val_batch_size(data_config, val_dataset_len: int) -> int:
    """Pick the SFT validation dataloader batch size.

    Matches PPO: ``data.val_batch_size`` if set, otherwise the full val set.
    ``micro_batch_size_per_gpu`` is an engine split size, not a dataloader knob.
    """
    val_batch_size = data_config.get("val_batch_size", None)
    if val_batch_size is None:
        val_batch_size = val_dataset_len
    return max(1, int(val_batch_size))


def sft_val_num_samples(batch) -> int:
    """Number of sequences in a collated SFT val batch."""
    batch_size = getattr(batch, "batch_size", None)
    if batch_size:
        return int(batch_size[0])
    if hasattr(batch, "__contains__") and "input_ids" in batch:
        return int(batch["input_ids"].shape[0])
    return 1


def reduce_sft_val_loss(losses_and_counts: list[tuple[float, int]]) -> Optional[float]:
    """Sample-weighted mean of per-batch val losses. None if there were no samples."""
    total_n = sum(n for _, n in losses_and_counts)
    if total_n <= 0:
        return None
    return sum(float(loss) * n for loss, n in losses_and_counts) / total_n
