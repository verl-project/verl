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

import os
import sys
import types

import pytest

from verl.utils import import_utils
from verl.utils.import_utils import get_trl_value_head_class, load_extern_object

# Path to the test module
TEST_MODULE_PATH = os.path.join(os.path.dirname(__file__), "_test_module.py")


def test_load_extern_object_class():
    """Test loading a class from an external file"""
    TestClass = load_extern_object(TEST_MODULE_PATH, "TestClass")

    # Verify the class was loaded correctly
    assert TestClass is not None
    assert TestClass.__name__ == "TestClass"

    # Test instantiation and functionality
    instance = TestClass()
    assert instance.value == "default"

    # Test with a custom value
    custom_instance = TestClass("custom")
    assert custom_instance.get_value() == "custom"


def test_load_extern_object_function():
    """Test loading a function from an external file"""
    test_function = load_extern_object(TEST_MODULE_PATH, "test_function")

    # Verify the function was loaded correctly
    assert test_function is not None
    assert callable(test_function)

    # Test function execution
    result = test_function()
    assert result == "test_function_result"


def test_load_extern_object_constant():
    """Test loading a constant from an external file"""
    constant = load_extern_object(TEST_MODULE_PATH, "TEST_CONSTANT")

    # Verify the constant was loaded correctly
    assert constant is not None
    assert constant == "test_constant_value"


def test_load_extern_object_nonexistent_file():
    """Test behavior when file doesn't exist"""
    with pytest.raises(FileNotFoundError):
        load_extern_object("/nonexistent/path.py", "SomeType")


def test_load_extern_object_nonexistent_type():
    """Test behavior when type doesn't exist in the file"""
    with pytest.raises(AttributeError):
        load_extern_object(TEST_MODULE_PATH, "NonExistentType")


def test_load_extern_object_none_path():
    """Test behavior when file path is None"""
    with pytest.raises(AttributeError):
        load_extern_object(None, "SomeType")


def test_load_extern_object_invalid_module():
    """Test behavior when module has syntax errors"""
    # Create a temporary file with syntax errors
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as temp_file:
        temp_file.write("This is not valid Python syntax :")
        temp_path = temp_file.name

    try:
        with pytest.raises(RuntimeError):
            load_extern_object(temp_path, "SomeType")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


class _ValueHead:
    pass


def _fake_trl(monkeypatch, *, experimental_ppo: bool, top_level: bool):
    """Install a fake ``trl`` in sys.modules; ``None`` entries make an import fail."""
    monkeypatch.setattr(import_utils, "is_trl_available", lambda: True)
    trl = types.ModuleType("trl")
    if top_level:
        trl.AutoModelForCausalLMWithValueHead = _ValueHead
    monkeypatch.setitem(sys.modules, "trl", trl)
    if experimental_ppo:
        ppo = types.ModuleType("trl.experimental.ppo")
        ppo.AutoModelForCausalLMWithValueHead = _ValueHead
        monkeypatch.setitem(sys.modules, "trl.experimental", types.ModuleType("trl.experimental"))
        monkeypatch.setitem(sys.modules, "trl.experimental.ppo", ppo)
    else:
        monkeypatch.setitem(sys.modules, "trl.experimental", None)
        monkeypatch.setitem(sys.modules, "trl.experimental.ppo", None)


def test_get_trl_value_head_class_without_trl(monkeypatch):
    monkeypatch.setattr(import_utils, "is_trl_available", lambda: False)
    assert get_trl_value_head_class() is None


def test_get_trl_value_head_class_from_experimental_ppo(monkeypatch):
    _fake_trl(monkeypatch, experimental_ppo=True, top_level=False)
    assert get_trl_value_head_class() is _ValueHead


def test_get_trl_value_head_class_from_top_level(monkeypatch):
    # TRL releases that exported the class at the top level
    _fake_trl(monkeypatch, experimental_ppo=False, top_level=True)
    assert get_trl_value_head_class() is _ValueHead


def test_get_trl_value_head_class_removed(monkeypatch):
    # TRL >= 1.13 ships neither location; callers must not crash on import
    _fake_trl(monkeypatch, experimental_ppo=False, top_level=False)
    assert get_trl_value_head_class() is None
