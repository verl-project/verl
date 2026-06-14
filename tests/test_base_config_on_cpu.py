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

import pytest

from verl.base_config import BaseConfig


@pytest.fixture
def base_config_mock():
    """Fixture to create a mock BaseConfig instance with test attributes."""
    mock_config = BaseConfig()
    mock_config.test_attr = "test_value"
    return mock_config


def test_getitem_success(base_config_mock):
    """Test __getitem__ with existing attribute (happy path)."""
    assert base_config_mock["test_attr"] == "test_value"


def test_getitem_nonexistent_attribute(base_config_mock):
    """Test __getitem__ with non-existent attribute (exception path 1)."""
    with pytest.raises(AttributeError):
        _ = base_config_mock["nonexistent_attr"]


def test_getitem_invalid_key_type(base_config_mock):
    """Test __getitem__ with invalid key type (exception path 2)."""
    with pytest.raises(TypeError):
        _ = base_config_mock[123]  # type: ignore
