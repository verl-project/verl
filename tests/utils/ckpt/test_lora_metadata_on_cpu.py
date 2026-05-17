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

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).parents[3] / "verl" / "utils" / "checkpoint" / "lora_metadata.py"
spec = importlib.util.spec_from_file_location("lora_metadata", MODULE_PATH)
lora_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lora_metadata)

LORA_TRAIN_META_FILENAME = lora_metadata.LORA_TRAIN_META_FILENAME
get_lora_train_meta = lora_metadata.get_lora_train_meta
load_lora_train_meta = lora_metadata.load_lora_train_meta
save_lora_train_meta = lora_metadata.save_lora_train_meta


def test_get_lora_train_meta_from_hf_model_config():
    model_config = {
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora": {"rank": 0, "alpha": 32},
    }

    assert get_lora_train_meta(model_config) == {
        "r": 64,
        "lora_alpha": 128,
        "task_type": "CAUSAL_LM",
    }


def test_get_lora_train_meta_from_megatron_lora_config():
    model_config = {
        "lora_rank": 0,
        "lora_alpha": 16,
        "lora": {"rank": 16, "alpha": 64},
    }

    assert get_lora_train_meta(model_config) == {
        "r": 16,
        "lora_alpha": 64,
        "task_type": "CAUSAL_LM",
    }


def test_get_lora_train_meta_returns_none_when_lora_disabled():
    model_config = {"lora_rank": 0, "lora": {"rank": 0}}

    assert get_lora_train_meta(model_config) is None


def test_save_and_load_lora_train_meta(tmp_path):
    model_config = SimpleNamespace(lora_rank=8, lora_alpha=24, task_type="SEQ_CLS")

    meta_path = save_lora_train_meta(model_config, tmp_path)

    assert meta_path == str(tmp_path / LORA_TRAIN_META_FILENAME)
    assert json.loads((tmp_path / LORA_TRAIN_META_FILENAME).read_text()) == {
        "r": 8,
        "lora_alpha": 24,
        "task_type": "SEQ_CLS",
    }
    assert load_lora_train_meta(tmp_path) == {
        "r": 8,
        "lora_alpha": 24,
        "task_type": "SEQ_CLS",
    }
