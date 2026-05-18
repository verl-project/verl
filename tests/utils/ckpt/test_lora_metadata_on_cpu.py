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
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _install_checkpoint_handler_stubs():
    verl_module = types.ModuleType("verl")
    verl_module.__path__ = []
    utils_module = types.ModuleType("verl.utils")
    utils_module.__path__ = []
    checkpoint_module = types.ModuleType("verl.utils.checkpoint")
    checkpoint_module.__path__ = []
    workers_module = types.ModuleType("verl.workers")
    workers_module.__path__ = []

    hdfs_io_module = types.ModuleType("verl.utils.hdfs_io")
    hdfs_io_module.makedirs = lambda *args, **kwargs: None
    hdfs_io_module.copy = lambda *args, **kwargs: True

    fs_module = types.ModuleType("verl.utils.fs")
    fs_module.local_mkdir_safe = lambda path: Path(path).mkdir(parents=True, exist_ok=True)

    single_controller_module = types.ModuleType("verl.single_controller")
    single_controller_module.WorkerGroup = object

    checkpoint_manager_module = types.ModuleType("verl.utils.checkpoint.checkpoint_manager")
    checkpoint_manager_module.find_latest_ckpt_path = lambda *args, **kwargs: None
    checkpoint_manager_module.get_checkpoint_tracker_filename = lambda path: str(
        Path(path) / "latest_checkpointed_iteration.txt"
    )

    logger_module = types.ModuleType("verl.utils.logger")
    logger_module.log_with_rank = lambda *args, **kwargs: None

    engine_module = types.ModuleType("verl.workers.engine")
    engine_module.BaseEngine = object

    sys.modules.update(
        {
            "verl": verl_module,
            "verl.utils": utils_module,
            "verl.utils.checkpoint": checkpoint_module,
            "verl.utils.hdfs_io": hdfs_io_module,
            "verl.utils.fs": fs_module,
            "verl.single_controller": single_controller_module,
            "verl.utils.checkpoint.checkpoint_manager": checkpoint_manager_module,
            "verl.utils.logger": logger_module,
            "verl.workers": workers_module,
            "verl.workers.engine": engine_module,
        }
    )


_install_checkpoint_handler_stubs()

MODULE_PATH = Path(__file__).parents[3] / "verl" / "utils" / "checkpoint" / "checkpoint_handler.py"
spec = importlib.util.spec_from_file_location("checkpoint_handler", MODULE_PATH)
checkpoint_handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint_handler)

LORA_TRAIN_META_FILENAME = checkpoint_handler.LORA_TRAIN_META_FILENAME
get_lora_train_meta = checkpoint_handler.get_lora_train_meta
save_lora_train_meta = checkpoint_handler.save_lora_train_meta


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


def test_save_lora_train_meta_writes_actor_checkpoint_metadata(tmp_path):
    model_config = SimpleNamespace(lora_rank=8, lora_alpha=24, task_type="SEQ_CLS")

    meta_path = save_lora_train_meta(model_config, tmp_path)

    assert meta_path == str(tmp_path / LORA_TRAIN_META_FILENAME)
    assert json.loads((tmp_path / LORA_TRAIN_META_FILENAME).read_text()) == {
        "r": 8,
        "lora_alpha": 24,
        "task_type": "SEQ_CLS",
    }
