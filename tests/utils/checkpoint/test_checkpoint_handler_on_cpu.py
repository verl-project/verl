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

import os

from verl.utils.checkpoint.checkpoint_handler import extract_step


def test_extract_step_uses_checkpoint_directory_not_parent():
    path = os.path.join("/tmp", "global_step_900", "archive", "global_step_12")

    assert extract_step(path) == 12


def test_extract_step_accepts_trailing_path_separator():
    path = os.path.join("/tmp", "run", "global_step_37", "")

    assert extract_step(path) == 37


def test_extract_step_rejects_non_checkpoint_directory():
    path = os.path.join("/tmp", "run", "global_step_37", "actor")

    assert extract_step(path) is None
