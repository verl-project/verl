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


def resolve_sft_val_batch_size(data_config, train_batch_size_per_dp: int, val_dataset_len: int) -> int:
    """Pick a validation batch size that does not swallow small val sets.

    Preference: ``data.val_batch_size`` > ``data.micro_batch_size_per_gpu`` >
    the full val set. Falls back to the train per-DP batch only if nothing else
    is available.
    """
    val_batch_size = data_config.get("val_batch_size", None)
    if val_batch_size is None:
        val_batch_size = data_config.get("micro_batch_size_per_gpu", None)
    if val_batch_size is None:
        val_batch_size = val_dataset_len if val_dataset_len > 0 else train_batch_size_per_dp
    return max(1, int(val_batch_size))
