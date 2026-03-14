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
"""Tests for validate_release_tags() — shared validation logic across all backends."""

from __future__ import annotations

import pytest

from verl.workers.rollout._tag_utils import validate_release_tags


class TestValidateReleaseTags:
    def test_none_returns_both(self):
        assert validate_release_tags(None) == {"kv_cache", "weights"}

    def test_weights_only(self):
        assert validate_release_tags(["weights"]) == {"weights"}

    def test_kv_cache_only(self):
        assert validate_release_tags(["kv_cache"]) == {"kv_cache"}

    def test_both_explicit(self):
        assert validate_release_tags(["kv_cache", "weights"]) == {"kv_cache", "weights"}

    def test_duplicates_deduplicated(self):
        assert validate_release_tags(["weights", "weights"]) == {"weights"}

    def test_unknown_tag_raises(self):
        with pytest.raises(ValueError, match="Unknown release tags"):
            validate_release_tags(["bogus"])

    def test_mixed_valid_and_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown release tags"):
            validate_release_tags(["weights", "bogus"])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_release_tags([])
