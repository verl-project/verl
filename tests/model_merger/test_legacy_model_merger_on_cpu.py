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

import pytest
import torch

from scripts.legacy_model_merger import FSDPModelMerger, MegatronModelMerger, ModelMergerConfig


def _write_model_config(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}), encoding="utf-8"
    )


@pytest.mark.parametrize(
    "relative_config_dir",
    ("hf_config_and_tokenizer", "huggingface", "model/huggingface"),
)
def test_megatron_merger_finds_checkpoint_model_config(tmp_path, relative_config_dir):
    config_dir = tmp_path / relative_config_dir
    _write_model_config(config_dir)

    merger = MegatronModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="megatron",
            local_dir=str(tmp_path),
            hf_model_config_path=str(tmp_path),
        )
    )

    assert merger.hf_model_config_path == str(config_dir)


def test_megatron_merger_respects_explicit_hf_model_path(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    explicit_config_dir = tmp_path / "model-config"
    _write_model_config(checkpoint_dir / "hf_config_and_tokenizer")
    _write_model_config(explicit_config_dir)
    _write_model_config(explicit_config_dir / "hf_config_and_tokenizer")

    merger = MegatronModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="megatron",
            local_dir=str(checkpoint_dir),
            hf_model_config_path=str(checkpoint_dir),
            hf_model_path=str(explicit_config_dir),
        )
    )

    assert merger.hf_model_config_path == str(explicit_config_dir)


def test_fsdp_merger_ignores_megatron_config_subdirectory(tmp_path):
    _write_model_config(tmp_path)
    _write_model_config(tmp_path / "hf_config_and_tokenizer")

    merger = FSDPModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=str(tmp_path),
            hf_model_config_path=str(tmp_path),
        )
    )

    assert merger.hf_model_config_path == str(tmp_path)


def test_megatron_merger_prefers_current_checkpoint_layout(tmp_path):
    legacy_config_dir = tmp_path / "hf_config_and_tokenizer"
    v1_config_dir = tmp_path / "huggingface"
    v2_config_dir = tmp_path / "model" / "huggingface"
    for config_dir in (legacy_config_dir, v1_config_dir, v2_config_dir):
        _write_model_config(config_dir)

    merger = MegatronModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="megatron",
            local_dir=str(tmp_path),
            hf_model_config_path=str(tmp_path),
        )
    )

    assert merger.hf_model_config_path == str(v2_config_dir)


def test_megatron_merger_prefers_v1_over_pre_v1_layout(tmp_path):
    legacy_config_dir = tmp_path / "hf_config_and_tokenizer"
    v1_config_dir = tmp_path / "huggingface"
    _write_model_config(legacy_config_dir)
    _write_model_config(v1_config_dir)

    merger = MegatronModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="megatron",
            local_dir=str(tmp_path),
            hf_model_config_path=str(tmp_path),
        )
    )

    assert merger.hf_model_config_path == str(v1_config_dir)


def test_megatron_merger_uses_checkpoint_root_as_fallback(tmp_path):
    _write_model_config(tmp_path)

    merger = MegatronModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="megatron",
            local_dir=str(tmp_path),
            hf_model_config_path=str(tmp_path),
        )
    )

    assert merger.hf_model_config_path == str(tmp_path)


def test_megatron_merger_loads_pre_dist_checkpoint_shards(monkeypatch, tmp_path):
    _write_model_config(tmp_path / "hf_config_and_tokenizer")
    shard_dir = tmp_path / "model" / "mp_rank_00"
    shard_dir.mkdir(parents=True)
    embedding = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    torch.save([{"embedding.word_embeddings.weight": embedding}], shard_dir / "model.pt")

    merger = MegatronModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="megatron",
            local_dir=str(tmp_path),
            hf_model_config_path=str(tmp_path),
        )
    )
    saved_state_dict = {}
    monkeypatch.setattr(merger, "save_hf_model_and_tokenizer", saved_state_dict.update)

    merger.merge_and_save()

    assert set(saved_state_dict) == {"model.embed_tokens.weight"}
    torch.testing.assert_close(saved_state_dict["model.embed_tokens.weight"], embedding)
