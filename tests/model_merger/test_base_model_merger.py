# Copyright 2026 Amazon.com Inc and/or its affiliates
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

import torch

from verl.model_merger.base_model_merger import BaseModelMerger


class DummyModelMerger(BaseModelMerger):
    def __init__(self, local_dir: str, target_dir: str):
        # save_lora_adapter only depends on local_dir/target_dir.
        self.config = SimpleNamespace(local_dir=local_dir, target_dir=target_dir)

    def merge_and_save(self):
        raise NotImplementedError

    def cleanup(self):
        pass


def _build_test_state_dict(rank: int) -> dict[str, torch.Tensor]:
    return {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.zeros(rank, 8),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.zeros(8, rank),
        "base_model.model.model.embed_tokens.base_layer.weight": torch.zeros(8, 8),
    }


def test_save_lora_adapter_preserves_lora_alpha_from_checkpoint(tmp_path):
    local_dir = tmp_path / "checkpoint"
    target_dir = tmp_path / "merged"
    source_lora_dir = local_dir / "lora_adapter"
    source_lora_dir.mkdir(parents=True)
    with open(source_lora_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump({"lora_alpha": 32}, f)

    merger = DummyModelMerger(local_dir=str(local_dir), target_dir=str(target_dir))
    state_dict = _build_test_state_dict(rank=8)
    lora_path = merger.save_lora_adapter(state_dict)

    assert lora_path == str(target_dir / "lora_adapter")
    with open(target_dir / "lora_adapter" / "adapter_config.json", encoding="utf-8") as f:
        merged_config = json.load(f)

    assert merged_config["lora_alpha"] == 32


def test_save_lora_adapter_falls_back_to_lora_rank_when_missing_alpha(tmp_path):
    local_dir = tmp_path / "checkpoint"
    target_dir = tmp_path / "merged"
    local_dir.mkdir(parents=True)

    merger = DummyModelMerger(local_dir=str(local_dir), target_dir=str(target_dir))
    state_dict = _build_test_state_dict(rank=8)
    merger.save_lora_adapter(state_dict)

    with open(target_dir / "lora_adapter" / "adapter_config.json", encoding="utf-8") as f:
        merged_config = json.load(f)

    assert merged_config["lora_alpha"] == 8
