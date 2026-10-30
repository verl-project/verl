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

from types import SimpleNamespace

import pytest
import torch

from verl.utils.veomni.frozen_ple import freeze_ple_tables, prepare_ple_storage
from verl.utils.veomni.ple_export import ple_export_shapes, trim_ple_export


def make_model():
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp")
    model.ple = torch.nn.Module()
    model.ple.ple_embedding = torch.nn.Module()
    embedding = model.ple.ple_embedding
    embedding.rows_per_checkpoint_shard = 5
    embedding.padded_rows_per_shard = 8
    embedding.split_ngram_parts = 1
    embedding.ngram_embedding = torch.nn.ModuleDict({"shard_0": torch.nn.Embedding(8, 3)})
    model.projection = torch.nn.Linear(3, 3)
    return model


def test_freeze_excludes_only_lookup_tables_from_optimizer_updates():
    model = make_model()
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    table.grad = torch.ones_like(table)
    before = table.detach().clone()
    projection = model.projection.weight.detach().clone()
    assert freeze_ple_tables(model) == (1, table.numel() * table.element_size())
    assert table.grad is None
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    for _ in range(2):
        model.projection(table).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.testing.assert_close(table, before, rtol=0, atol=0)
    assert not torch.equal(projection, model.projection.weight)
    assert all(table is not p for group in optimizer.param_groups for p in group["params"])


@pytest.mark.parametrize("dtype", [None, "float32", "bfloat16"])
def test_storage_conversion_preserves_parameter_identity_and_trainable_weights(dtype):
    model = make_model()
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    before = table.detach().clone()
    prepare_ple_storage(model, dtype)
    assert model.ple.ple_embedding.ngram_embedding["shard_0"].weight is table
    expected_dtype = torch.float32 if dtype is None else getattr(torch, dtype)
    torch.testing.assert_close(table, before.to(expected_dtype))
    assert model.projection.weight.dtype == torch.float32


def test_freeze_rejects_wrong_model_and_missing_tables():
    model = make_model()
    model.config.model_type = "other"
    with pytest.raises(ValueError, match="only qwen4_exp"):
        freeze_ple_tables(model)
    model.config.model_type = "qwen4_exp"
    del model.ple
    with pytest.raises(ValueError, match="no PLE table"):
        freeze_ple_tables(model)


def test_trainable_ple_export_trims_global_padding_and_rejects_local_shards():
    model = make_model()
    name = "ple.ple_embedding.ngram_embedding.shard_0.weight"
    shapes = ple_export_shapes(model)
    values = torch.arange(24).reshape(8, 3)
    exported = trim_ple_export(name, values, shapes)
    torch.testing.assert_close(exported, values[:5])
    assert exported.data_ptr() == values.data_ptr()
    with pytest.raises(ValueError, match="Invalid PLE export shape"):
        trim_ple_export(name, values[:4], shapes)
