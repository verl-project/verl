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

"""CPU tests for the router replay batch-flag guard."""

import pytest

from verl.workers.engine.megatron.transformer_impl import _validate_router_replay_batch_flag


def test_guard_allows_disabled_engine():
    _validate_router_replay_batch_flag(False, False)


def test_guard_allows_enabled_engine_with_flag():
    _validate_router_replay_batch_flag(True, True)


def test_guard_rejects_missing_batch_flag():
    with pytest.raises(RuntimeError, match="missing the 'enable_routing_replay' flag"):
        _validate_router_replay_batch_flag(True, False)
