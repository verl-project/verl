# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import subprocess
import sys
from pathlib import Path


def test_checkpoint_view_updates_code_without_modifying_source(tmp_path):
    source, target, code = (tmp_path / name for name in ("source", "view", "code"))
    source.mkdir()
    code.mkdir()
    (source / "config.json").write_text('{"model_type": "kimi_k3"}')
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "modeling_kimi_k3.py").write_text("original")
    (code / "modeling_kimi_k3.py").write_text("from .kimi_k3 import iter_checkpoint_weights")
    (code / "kimi_k3").mkdir()
    (code / "kimi_k3/__init__.py").write_text("from .checkpoint import iter_checkpoint_weights")
    (code / "kimi_k3/checkpoint.py").write_text("def iter_checkpoint_weights(): pass")
    script = Path(__file__).resolve().parents[2] / "examples/ascend_extras/grpo_trainer/prepare_kimi_k3_model.py"
    command = [sys.executable, str(script), str(source), str(target), "--code", str(code)]
    subprocess.run(command, check=True)
    subprocess.run(command, check=True)
    assert (target / "model.safetensors").resolve() == source / "model.safetensors"
    assert not (target / "config.json").is_symlink()
    import json

    from transformers.dynamic_module_utils import get_relative_import_files

    assert json.loads((target / "config.json").read_text())["load_checkpoint_after_shard"]
    assert "load_checkpoint_after_shard" not in json.loads((source / "config.json").read_text())
    imports = get_relative_import_files(target / "modeling_kimi_k3.py")
    assert {Path(name).name for name in imports} == {"kimi_k3.py"}
    assert (source / "modeling_kimi_k3.py").read_text() == "original"

    command[3] = str(source)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Model view must be separate" in result.stderr
