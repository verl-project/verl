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

from types import SimpleNamespace

import pytest
import torch

from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def _merger(state_dict):
    merger = object.__new__(FSDPModelMerger)
    merger.config = SimpleNamespace(operation="merge", lora_only=True, hf_upload=False)
    merger._get_world_size = lambda: 2
    merger._load_rank_zero_state_dict = lambda world_size: state_dict
    merger._extract_device_mesh_info = lambda state, world_size: (
        torch.arange(2),
        ("fsdp",),
    )
    merger._calculate_shard_configuration = lambda mesh, names: (2, (2,))
    merger._load_and_merge_state_dicts = lambda *args: dict(state_dict)
    return merger


def test_lora_only_export_skips_full_model_save(tmp_path):
    state_dict = {
        "base_model.model.layers.0.q_proj.lora_A.default.weight": torch.ones(2, 4),
        "base_model.model.layers.0.q_proj.lora_B.default.weight": torch.ones(4, 2),
    }
    merger = _merger(state_dict)
    merger.save_lora_adapter = lambda state: str(tmp_path / "lora_adapter")
    merger.save_hf_model_and_tokenizer = lambda state: pytest.fail("full model save must not run")

    merger.merge_and_save()


def test_lora_only_export_rejects_base_weights():
    merger = _merger(
        {
            "base_model.model.layers.0.q_proj.lora_A.default.weight": torch.ones(2, 4),
            "model.layers.0.weight": torch.ones(2, 2),
        }
    )

    with pytest.raises(ValueError, match="adapter-only checkpoint"):
        merger.merge_and_save()
