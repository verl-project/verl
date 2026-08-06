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

import pytest

from verl.utils.import_utils import load_extern_object

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
