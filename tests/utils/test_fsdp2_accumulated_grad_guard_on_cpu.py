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

"""The FSDP2 `_unsharded_param` guard lives in the shared FSDP utility.

Every FSDP2 wrapper in the tree reaches `apply_fsdp2`, so the guard is installed
from there rather than from one engine module that a direct caller of
`apply_fsdp2` (the recipes, the reward-model workers) would never import.
"""

import pytest

pytest.importorskip("torch.distributed.fsdp._fully_shard._fsdp_param")

from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

from verl.utils.fsdp_utils import _guard_fsdp2_accumulated_grad


def test_a_param_that_was_never_gathered_no_longer_raises():
    _guard_fsdp2_accumulated_grad()

    # A parameter that never took part in the forward pass has no
    # `_unsharded_param`; the unguarded method dereferences it and raises.
    param = FSDPParam.__new__(FSDPParam)
    assert not hasattr(param, "_unsharded_param")
    assert param.to_accumulated_grad_if_needed() is None


def test_the_guard_is_installed_once():
    _guard_fsdp2_accumulated_grad()
    first = FSDPParam.to_accumulated_grad_if_needed
    _guard_fsdp2_accumulated_grad()
    assert FSDPParam.to_accumulated_grad_if_needed is first
    assert getattr(first, "_verl_guarded", False)
