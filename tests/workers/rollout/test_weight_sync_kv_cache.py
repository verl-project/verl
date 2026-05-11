# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
omegaconf = pytest.importorskip("omegaconf")
DictConfig = omegaconf.DictConfig

from verl.workers.rollout.weight_sync_kv_cache import should_flush_kv_cache_after_weight_sync


def test_should_flush_default_and_none():
    assert should_flush_kv_cache_after_weight_sync(None) is True
    assert should_flush_kv_cache_after_weight_sync(SimpleNamespace()) is True


def test_should_flush_respects_stale_kv_flag():
    assert should_flush_kv_cache_after_weight_sync(SimpleNamespace(allow_stale_kv_cache_after_weight_sync=False)) is True
    assert should_flush_kv_cache_after_weight_sync(SimpleNamespace(allow_stale_kv_cache_after_weight_sync=True)) is False


def test_should_flush_omegaconf_missing_key():
    cfg = DictConfig({})
    assert should_flush_kv_cache_after_weight_sync(cfg) is True


def test_should_flush_omegaconf_stale_kv():
    cfg = DictConfig({"allow_stale_kv_cache_after_weight_sync": True})
    assert should_flush_kv_cache_after_weight_sync(cfg) is False
