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
import unittest
from unittest.mock import patch

from verl.utils.tracking import mlflow_experiment_name


class TestMlflowExperimentName(unittest.TestCase):
    def test_no_prefix_returns_bare_name(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mlflow_experiment_name("my_project"), "my_project")

    def test_empty_prefix_is_noop(self):
        with patch.dict(os.environ, {"VERL_MLFLOW_EXPERIMENT_PREFIX": ""}):
            self.assertEqual(mlflow_experiment_name("my_project"), "my_project")

    def test_prefix_nests_bare_name(self):
        with patch.dict(os.environ, {"VERL_MLFLOW_EXPERIMENT_PREFIX": "/Workspace/Shared/mlflow"}):
            self.assertEqual(mlflow_experiment_name("my_project"), "/Workspace/Shared/mlflow/my_project")

    def test_unrooted_name_with_slash_is_nested(self):
        # Only a leading slash marks a name as already rooted; inner slashes do not.
        with patch.dict(os.environ, {"VERL_MLFLOW_EXPERIMENT_PREFIX": "/Workspace/Shared/mlflow"}):
            self.assertEqual(mlflow_experiment_name("team/my_project"), "/Workspace/Shared/mlflow/team/my_project")

    def test_trailing_slash_is_normalized(self):
        with patch.dict(os.environ, {"VERL_MLFLOW_EXPERIMENT_PREFIX": "/root/"}):
            self.assertEqual(mlflow_experiment_name("p"), "/root/p")

    def test_absolute_name_is_left_untouched(self):
        with patch.dict(os.environ, {"VERL_MLFLOW_EXPERIMENT_PREFIX": "/root"}):
            self.assertEqual(mlflow_experiment_name("/Users/me/exp"), "/Users/me/exp")


if __name__ == "__main__":
    unittest.main()
