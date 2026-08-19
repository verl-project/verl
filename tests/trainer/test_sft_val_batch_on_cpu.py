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

from verl.trainer.sft_val_utils import resolve_sft_val_batch_size


def test_resolve_prefers_explicit_val_batch_size():
    assert resolve_sft_val_batch_size({"val_batch_size": 16, "micro_batch_size_per_gpu": 4}, 256, 200) == 16


def test_resolve_falls_back_to_micro_batch_not_train_batch():
    assert resolve_sft_val_batch_size({"micro_batch_size_per_gpu": 4}, 256, 200) == 4


def test_resolve_uses_val_len_when_no_micro_batch():
    assert resolve_sft_val_batch_size({}, 256, 200) == 200


def test_small_val_set_is_not_empty_with_resolved_batch():
    train_bs = 256
    n = 200
    assert n // train_bs == 0  # old train-batch + drop_last path
    batch = resolve_sft_val_batch_size({"micro_batch_size_per_gpu": 4}, train_bs, n)
    assert (n + batch - 1) // batch > 0
