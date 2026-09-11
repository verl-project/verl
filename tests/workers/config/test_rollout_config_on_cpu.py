# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from verl.workers.config.rollout import CheckpointEngineConfig


@pytest.mark.parametrize("value", [-1, 1.5, "0", None])
def test_weight_transfer_gc_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="gc_on_weight_transfer_cleanup must be a boolean or non-negative integer"):
        CheckpointEngineConfig(gc_on_weight_transfer_cleanup=value)
