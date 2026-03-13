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

import os

import pytest

import verl.workers.config.model as model_config_module
from verl.workers.config.model import DiffusersModelConfig


@pytest.fixture(autouse=True)
def patch_model_io(monkeypatch):
    monkeypatch.setattr(model_config_module, "copy_to_local", lambda path, use_shm=False: path)
    monkeypatch.setattr(model_config_module, "import_external_libs", lambda external_lib: None)


class TestDiffusersModelConfigCPU:
    def test_default_tokenizer_path_and_local_path(self):
        config = DiffusersModelConfig(path="/tmp/fake-model", load_tokenizer=False)
        assert config.local_path == "/tmp/fake-model"
        assert config.tokenizer_path == os.path.join("/tmp/fake-model", "tokenizer")

    def test_target_modules_accepts_list_of_strings(self):
        config = DiffusersModelConfig(
            path="/tmp/fake-model",
            load_tokenizer=False,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        assert config.target_modules == ["q_proj", "k_proj", "v_proj"]

    def test_target_modules_raises_on_invalid_list_elements(self):
        with pytest.raises(TypeError, match="All elements in target_modules list must be strings"):
            DiffusersModelConfig(
                path="/tmp/fake-model",
                load_tokenizer=False,
                target_modules=["q_proj", 1],
            )
