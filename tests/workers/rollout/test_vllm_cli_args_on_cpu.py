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

import json

import pytest

from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config


class TestBuildCliArgsFromConfig:
    """Tests for CLI argument serialization from config dictionaries."""

    def test_string_value(self):
        """String values become '--key value'."""
        config = {"model": "gpt2"}
        result = build_cli_args_from_config(config)
        assert result == ["--model", "gpt2"]

    def test_integer_value(self):
        """Integer values are converted to strings."""
        config = {"tensor-parallel-size": 4}
        result = build_cli_args_from_config(config)
        assert result == ["--tensor-parallel-size", "4"]

    def test_float_value(self):
        """Float values are converted to strings."""
        config = {"temperature": 0.7}
        result = build_cli_args_from_config(config)
        assert result == ["--temperature", "0.7"]

    def test_bool_true(self):
        """Bool True adds flag without value."""
        config = {"enable-prefix-caching": True}
        result = build_cli_args_from_config(config)
        assert result == ["--enable-prefix-caching"]

    def test_bool_false(self):
        """Bool False is skipped entirely."""
        config = {"enable-prefix-caching": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_none_value(self):
        """None values are skipped."""
        config = {"lora-path": None}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_values(self):
        """List values are expanded into multiple arguments."""
        config = {"cudagraph-capture-sizes": [1, 2, 4, 8]}
        result = build_cli_args_from_config(config)
        assert result == ["--cudagraph-capture-sizes", "1", "2", "4", "8"]

    def test_empty_list(self):
        """Empty lists are skipped (vLLM nargs='+' requires at least one value)."""
        config = {"cudagraph-capture-sizes": []}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_with_strings(self):
        """List of strings is properly expanded."""
        config = {"allowed-origins": ["http://localhost", "http://example.com"]}
        result = build_cli_args_from_config(config)
        assert result == ["--allowed-origins", "http://localhost", "http://example.com"]

    def test_dict_value(self):
        """Dict values are JSON serialized."""
        config = {"extra-config": {"key": "value", "nested": True}}
        result = build_cli_args_from_config(config)
        assert result[0] == "--extra-config"
        # JSON output may have different key ordering, so parse and compare
        assert json.loads(result[1]) == {"key": "value", "nested": True}

    def test_mixed_config(self):
        """Test a realistic mixed configuration."""
        config = {
            "tensor-parallel-size": 4,
            "enable-prefix-caching": True,
            "disable-log-requests": False,
            "lora-path": None,
            "cudagraph-capture-sizes": [1, 2, 4, 8],
            "max-model-len": 2048,
        }
        result = build_cli_args_from_config(config)

        # Check expected args are present
        assert "--tensor-parallel-size" in result
        assert "4" in result
        assert "--enable-prefix-caching" in result
        assert "--cudagraph-capture-sizes" in result
        assert "1" in result
        assert "8" in result
        assert "--max-model-len" in result
        assert "2048" in result

        # Check skipped values are not present
        assert "--disable-log-requests" not in result
        assert "--lora-path" not in result

    def test_preserves_order(self):
        """Arguments should preserve dictionary order (Python 3.7+)."""
        config = {"first": "a", "second": "b", "third": "c"}
        result = build_cli_args_from_config(config)
        assert result == ["--first", "a", "--second", "b", "--third", "c"]

    def test_empty_config(self):
        """Empty config returns empty list."""
        config = {}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_single_element_list(self):
        """Single element list works correctly."""
        config = {"sizes": [42]}
        result = build_cli_args_from_config(config)
        assert result == ["--sizes", "42"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
