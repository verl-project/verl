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

import pytest

# Assuming REWARD_MANAGER_REGISTRY is defined somewhere in the module
from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY, get_reward_manager_cls, register


@pytest.fixture
def setup():
    """Setup test cases with a mock registry."""
    REWARD_MANAGER_REGISTRY.clear()
    REWARD_MANAGER_REGISTRY.update({"manager1": "Manager1Class", "manager2": "Manager2Class"})
    return REWARD_MANAGER_REGISTRY


def test_get_existing_manager(setup):
    """Test getting an existing reward manager class."""
    assert get_reward_manager_cls("manager1") == "Manager1Class"
    assert get_reward_manager_cls("manager2") == "Manager2Class"


def test_get_nonexistent_manager(setup):
    """Test getting a non-existent reward manager raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        get_reward_manager_cls("unknown_manager")
    assert "Unknown reward manager: unknown_manager" in str(excinfo.value)


def test_case_sensitivity(setup):
    """Test that manager names are case-sensitive."""
    with pytest.raises(ValueError):
        get_reward_manager_cls("MANAGER1")
    with pytest.raises(ValueError):
        get_reward_manager_cls("Manager1")


def test_empty_registry(setup):
    """Test behavior when registry is empty."""
    REWARD_MANAGER_REGISTRY.clear()
    with pytest.raises(ValueError) as excinfo:
        get_reward_manager_cls("any_manager")
    assert "Unknown reward manager: any_manager" in str(excinfo.value)


def test_register_new_class(setup):
    """Test registering a new class with the decorator."""

    @register("test_manager")
    class TestManager:
        pass

    assert "test_manager" in REWARD_MANAGER_REGISTRY
    assert REWARD_MANAGER_REGISTRY["test_manager"] == TestManager


def test_register_different_classes_same_name(setup):
    """Test that registering different classes with same name raises ValueError."""

    @register("conflict_manager")
    class Manager1:
        pass

    with pytest.raises(ValueError):

        @register("conflict_manager")
        class Manager2:
            pass

    assert REWARD_MANAGER_REGISTRY["conflict_manager"] == Manager1


def test_decorator_returns_original_class(setup):
    """Test that the decorator returns the original class unchanged."""

    @register("return_test")
    class OriginalClass:
        def method(setup):
            return 42

    assert OriginalClass().method() == 42
    assert REWARD_MANAGER_REGISTRY["return_test"] == OriginalClass
