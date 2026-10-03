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

import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, Qwen2Config

from verl.model_merger import base_model_merger
from verl.model_merger.base_model_merger import BaseModelMerger
from verl.model_merger.output_validation import validate_hf_model_output


class _TestModelMerger(BaseModelMerger):
    def merge_and_save(self):
        pass

    def cleanup(self):
        pass


class _ConfigOnlyModel:
    def to_empty(self, device):
        return self

    def can_generate(self):
        return False

    def save_pretrained(self, target_dir, state_dict):
        with open(target_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"model_type": "test"}, f)


class _ConfigOnlyAutoModel:
    @classmethod
    def from_config(cls, model_config, **kwargs):
        return _ConfigOnlyModel()


class _FakeTokenizer:
    def save_pretrained(self, target_dir):
        with open(target_dir / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump({"version": "1.0"}, f)


def test_save_rejects_tokenizer_only_output(monkeypatch, tmp_path):
    merger = _TestModelMerger.__new__(_TestModelMerger)
    merger.config = SimpleNamespace(target_dir=tmp_path, trust_remote_code=False, local_dir=None)
    merger.hf_model_config_path = tmp_path
    merger.model_config = SimpleNamespace()

    monkeypatch.setattr(merger, "get_transformers_auto_model_class", lambda: _ConfigOnlyAutoModel)
    monkeypatch.setattr(base_model_merger, "drop_tied_target_keys", lambda *args: None)
    monkeypatch.setattr(base_model_merger, "hf_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(base_model_merger, "hf_tokenizer", lambda *args, **kwargs: _FakeTokenizer())

    with pytest.raises(RuntimeError, match="no complete model weights"):
        merger.save_hf_model_and_tokenizer({"weight": torch.ones(1)})


def test_save_accepts_actual_hf_model_output(monkeypatch, tmp_path):
    model_config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    source_model = AutoModelForCausalLM.from_config(model_config)

    merger = _TestModelMerger.__new__(_TestModelMerger)
    merger.config = SimpleNamespace(target_dir=tmp_path, trust_remote_code=False, local_dir=None)
    merger.hf_model_config_path = tmp_path
    merger.model_config = model_config

    monkeypatch.setattr(merger, "get_transformers_auto_model_class", lambda: AutoModelForCausalLM)
    monkeypatch.setattr(merger, "patch_model_generation_config", lambda model: model)
    monkeypatch.setattr(base_model_merger, "hf_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(base_model_merger, "hf_tokenizer", lambda *args, **kwargs: None)

    merger.save_hf_model_and_tokenizer(source_model.state_dict())

    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "model.safetensors").is_file()


def test_save_generation_config_preserves_pretrained_values(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    GenerationConfig(
        eos_token_id=[151643, 151645],
        max_new_tokens=4096,
    ).save_pretrained(source_dir)

    merger = _TestModelMerger.__new__(_TestModelMerger)
    merger.hf_model_config_path = source_dir
    merger.save_generation_config(target_dir)

    saved_config = GenerationConfig.from_pretrained(target_dir)
    assert saved_config.eos_token_id == [151643, 151645]
    assert saved_config.max_new_tokens == 4096


def test_save_generation_config_allows_missing_file(capsys, tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    stale_config_path = target_dir / "generation_config.json"
    stale_config_path.write_text('{"eos_token_id": 0}', encoding="utf-8")

    merger = _TestModelMerger.__new__(_TestModelMerger)
    merger.hf_model_config_path = source_dir
    merger.save_generation_config(target_dir)

    assert not stale_config_path.exists()
    assert capsys.readouterr().out == ""


def test_load_generation_config_warns_when_file_is_missing(capsys, tmp_path):
    merger = _TestModelMerger.__new__(_TestModelMerger)
    merger.hf_model_config_path = tmp_path

    assert merger.load_generation_config() is None
    assert f"Warning: Generation config file not found in {tmp_path}" in capsys.readouterr().out


def test_save_generation_config_rejects_invalid_source(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    (source_dir / "generation_config.json").write_text("{invalid", encoding="utf-8")
    stale_config_path = target_dir / "generation_config.json"
    stale_config_path.write_text('{"eos_token_id": 0}', encoding="utf-8")

    merger = _TestModelMerger.__new__(_TestModelMerger)
    merger.hf_model_config_path = source_dir

    with pytest.raises(OSError, match="not a valid JSON file"):
        merger.save_generation_config(target_dir)

    assert stale_config_path.is_file()


@pytest.mark.parametrize("weight_name", ("model.safetensors", "pytorch_model.bin"))
def test_validate_hf_model_output_accepts_single_weight_file(tmp_path, weight_name):
    (tmp_path / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    (tmp_path / weight_name).write_bytes(b"weights")

    validate_hf_model_output(tmp_path)


@pytest.mark.parametrize(
    ("index_name", "shard_names"),
    (
        ("model.safetensors.index.json", ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")),
        ("pytorch_model.bin.index.json", ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin")),
    ),
)
def test_validate_hf_model_output_accepts_complete_sharded_weights(tmp_path, index_name, shard_names):
    (tmp_path / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    weight_map = {f"layer.{index}.weight": shard_name for index, shard_name in enumerate(shard_names)}
    (tmp_path / index_name).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    for shard_name in shard_names:
        (tmp_path / shard_name).write_bytes(b"weights")

    validate_hf_model_output(tmp_path)


def test_validate_hf_model_output_rejects_missing_shard(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    index = {
        "weight_map": {
            "layer.0.weight": "model-00001-of-00002.safetensors",
            "layer.1.weight": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="model-00002-of-00002.safetensors"):
        validate_hf_model_output(tmp_path)


def test_validate_hf_model_output_accepts_single_file_with_stale_index(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    index = {"weight_map": {"layer.0.weight": "model-00001-of-00001.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    validate_hf_model_output(tmp_path)


def test_validate_hf_model_output_rejects_escaping_shard_path(tmp_path):
    target_dir = tmp_path / "output"
    target_dir.mkdir()
    (target_dir / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    (tmp_path / "external.safetensors").write_bytes(b"weights")
    index = {"weight_map": {"layer.0.weight": "../external.safetensors"}}
    (target_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsafe shard paths"):
        validate_hf_model_output(target_dir)


def test_validate_hf_model_output_rejects_non_string_shard_name(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    index = {"weight_map": {"layer.0.weight": []}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid shard names"):
        validate_hf_model_output(tmp_path)


def test_validate_hf_model_output_rejects_missing_config(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="missing or empty config.json"):
        validate_hf_model_output(tmp_path)


def test_validate_hf_model_output_rejects_empty_weight_file(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    (tmp_path / "model.safetensors").touch()

    with pytest.raises(RuntimeError, match="no complete model weights"):
        validate_hf_model_output(tmp_path)
