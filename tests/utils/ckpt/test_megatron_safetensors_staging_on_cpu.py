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

from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors
import safetensors.torch as safetensors_torch
import torch
from safetensors import safe_open
from safetensors.torch import save_file as imported_save_file

from verl.utils.checkpoint import megatron_checkpoint_manager as checkpoint_module


@pytest.mark.parametrize("installed_version", ["0.7.0", "0.8.0"])
def test_imported_save_file_roundtrips_data_and_metadata(monkeypatch, tmp_path, installed_version):
    original_serialize_file = safetensors_torch.serialize_file
    writes = []

    def record_serialize(data, filename, metadata=None):
        writes.append(Path(filename))
        original_serialize_file(data, filename, metadata=metadata)

    monkeypatch.setattr(safetensors, "__version__", installed_version)
    monkeypatch.setattr(safetensors_torch, "serialize_file", record_serialize)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)

    def save_weights(models, weights_path):
        for shard in range(2):
            destination = Path(weights_path) / f"model-{shard}.safetensors"
            imported_save_file({"weight": tensor + shard}, destination, metadata={"format": "pt"})
            with safe_open(destination, framework="pt") as archive:
                torch.testing.assert_close(archive.get_tensor("weight"), tensor + shard)
                assert archive.metadata() == {"format": "pt"}
            if installed_version == "0.7.0":
                assert writes[-1] == destination
            else:
                assert writes[-1].parent.parent == Path("/tmp")
                assert not writes[-1].parent.exists()

    manager = object.__new__(checkpoint_module.MegatronCheckpointManager)
    manager.model = object()
    manager.vanilla_bridge = True
    manager.checkpoint_config = SimpleNamespace(mbridge_config={})
    manager.bridge = SimpleNamespace(save_weights=save_weights)
    manager._save_model_as_hf_via_bridge(str(tmp_path))
    assert safetensors_torch.serialize_file is record_serialize


@pytest.mark.parametrize("failed_stage", ["serialize", "copy"])
def test_staging_failure_propagates_restores_serializer_and_cleans_temp(monkeypatch, tmp_path, failed_stage):
    original_serialize_file = safetensors_torch.serialize_file
    writes = []

    def serialize(data, filename, metadata=None):
        writes.append(Path(filename))
        if failed_stage == "serialize":
            raise OSError("serialize failed")
        original_serialize_file(data, filename, metadata=metadata)

    def fail_copy(source, destination):
        raise OSError("copy failed")

    def save_weights(models, weights_path):
        imported_save_file({"weight": torch.ones(2)}, Path(weights_path) / "model.safetensors")

    monkeypatch.setattr(safetensors, "__version__", "0.8.0")
    monkeypatch.setattr(safetensors_torch, "serialize_file", serialize)
    monkeypatch.setattr(checkpoint_module.shutil, "copyfile", fail_copy)
    manager = object.__new__(checkpoint_module.MegatronCheckpointManager)
    manager.model = object()
    manager.vanilla_bridge = True
    manager.checkpoint_config = SimpleNamespace(mbridge_config={})
    manager.bridge = SimpleNamespace(save_weights=save_weights)
    with pytest.raises(OSError, match=f"{failed_stage} failed"):
        manager._save_model_as_hf_via_bridge(str(tmp_path))

    assert safetensors_torch.serialize_file is serialize
    assert len(writes) == 1
    assert not writes[0].parent.exists()


@pytest.mark.parametrize("bridge_kind", ["vanilla", "hf", "adapter"])
def test_hf_bridge_keeps_original_destination_and_participating_rank(monkeypatch, tmp_path, bridge_kind):
    monkeypatch.setattr(safetensors, "__version__", "0.8.0")
    original_serialize_file = safetensors_torch.serialize_file
    calls = []
    model = object()
    peft = object() if bridge_kind == "adapter" else None

    def save_weights(models, weights_path, memory_efficient=False):
        assert safetensors_torch.serialize_file is not original_serialize_file
        assert models is model
        calls.append((Path(weights_path), memory_efficient))
        Path(weights_path).mkdir(exist_ok=True)
        imported_save_file({"weight": torch.ones(2)}, Path(weights_path) / "model.safetensors")

    def save_hf_weights(models, weights_path, *, strict):
        assert strict is True
        save_weights(models, weights_path)

    def save_hf_adapter(models, weights_path, peft_cls):
        assert peft_cls is peft
        save_weights(models, weights_path)

    manager = object.__new__(checkpoint_module.MegatronCheckpointManager)
    manager.model = model
    manager.rank = 1
    manager.vanilla_bridge = bridge_kind == "vanilla"
    manager.peft_cls = peft
    manager.checkpoint_config = SimpleNamespace(strict=True, mbridge_config={"memory_efficient": True})
    manager.bridge = SimpleNamespace(
        save_weights=save_weights, save_hf_weights=save_hf_weights, save_hf_adapter=save_hf_adapter
    )
    (tmp_path / "tokenizer.json").write_text("existing tokenizer")

    manager._save_model_as_hf_via_bridge(str(tmp_path))

    expected_path = tmp_path / "adapter" if peft else tmp_path
    assert calls == [(expected_path, bridge_kind == "vanilla")]
    assert (expected_path / "model.safetensors").is_file()
    assert (tmp_path / "tokenizer.json").read_text() == "existing tokenizer"
    assert safetensors_torch.serialize_file is original_serialize_file


def test_bridge_failure_restores_serializer(monkeypatch, tmp_path):
    monkeypatch.setattr(safetensors, "__version__", "0.8.0")
    original_serialize_file = safetensors_torch.serialize_file

    def save_weights(models, weights_path):
        assert safetensors_torch.serialize_file is not original_serialize_file
        raise RuntimeError("bridge failed")

    manager = object.__new__(checkpoint_module.MegatronCheckpointManager)
    manager.model = object()
    manager.vanilla_bridge = True
    manager.checkpoint_config = SimpleNamespace(mbridge_config={})
    manager.bridge = SimpleNamespace(save_weights=save_weights)

    with pytest.raises(RuntimeError, match="bridge failed"):
        manager._save_model_as_hf_via_bridge(str(tmp_path))
    assert safetensors_torch.serialize_file is original_serialize_file
