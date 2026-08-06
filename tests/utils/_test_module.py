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


# Test module for import_utils.load_extern_object testing
class TestClass:
    """A test class to be imported by load_extern_object"""

    def __init__(self, value=None):
        self.value = value or "default"

    def get_value(self):
        return self.value


TEST_CONSTANT = "test_constant_value"


def test_function():
    return "test_function_result"
