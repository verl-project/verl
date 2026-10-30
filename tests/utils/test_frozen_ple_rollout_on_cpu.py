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

from verl.utils.veomni.frozen_ple import validate_frozen_ple_rollout


@pytest.mark.parametrize("load_format", ["dummy", "dummy_hf", "pt"])
def test_frozen_ple_rejects_unverified_rollout_initialization(load_format):
    with pytest.raises(ValueError, match="excludes PLE tables from actor-to-rollout weight sync"):
        validate_frozen_ple_rollout(True, "vllm", load_format)


@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
def test_frozen_ple_accepts_checkpoint_initialization(load_format):
    validate_frozen_ple_rollout(True, "vllm", load_format)


def test_trainable_ple_can_still_use_dummy_initialization():
    validate_frozen_ple_rollout(False, "vllm", "dummy")
