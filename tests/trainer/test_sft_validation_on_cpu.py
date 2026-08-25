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

import pytest

from verl.trainer.sft_trainer import _should_run_validation


@pytest.mark.parametrize(
    ("has_val_dataloader", "is_last_step", "global_step", "test_freq", "expected"),
    [
        (False, False, 2, 1, False),
        (False, True, 2, 1, False),
        (True, False, 2, 1, True),
        (True, False, 2, 3, False),
        (True, True, 2, -1, True),
        (True, False, 2, 0, False),
    ],
)
def test_should_run_validation(has_val_dataloader, is_last_step, global_step, test_freq, expected):
    val_dataloader = object() if has_val_dataloader else None

    assert _should_run_validation(val_dataloader, is_last_step, global_step, test_freq) is expected
